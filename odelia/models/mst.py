
import math
import torch 
import torch.nn as nn 
import torchvision.models as models
from einops import rearrange
from torch.utils.checkpoint import checkpoint
from x_transformers import Encoder
from transformers import AutoModel, DINOv3ViTModel, Dinov2WithRegistersModel

from .base_model import BasicClassifier, BasicRegression

def _get_resnet_torch(model):
    return {
        18: models.resnet18, 34: models.resnet34, 50: models.resnet50, 101: models.resnet101, 152: models.resnet152
    }.get(model) 



class _MST(nn.Module):
    def __init__(
        self, 
        out_ch=1, 
        backbone_type="dinov3",
        model_size = "s", # 34, 50, ... or 's', 'b', 'l'
        slice_fusion_type = "transformer", # transformer, linear, average, none 
    ):
        super().__init__()
        self.backbone_type = backbone_type
        self.slice_fusion_type = slice_fusion_type

        if backbone_type == "resnet":
            Model = _get_resnet_torch(model_size)
            self.backbone = Model(weights="DEFAULT")
            emb_ch = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone_type == "dinov2":
            model_size = {'s':'small', 'b':'base', 'l':'large'}.get(model_size)
            self.backbone = Dinov2WithRegistersModel.from_pretrained(f"facebook/dinov2-with-registers-{model_size}")
            emb_ch = self.backbone.config.hidden_size
        elif backbone_type == "dinov3":
            self.backbone = DINOv3ViTModel.from_pretrained(f"facebook/dinov3-vit{model_size}16-pretrain-lvd1689m")
            emb_ch = self.backbone.config.hidden_size
        else:
            raise ValueError("Unknown backbone_type")


        self.emb_ch = emb_ch 
        if slice_fusion_type == "transformer":
            self.slice_fusion = Encoder(
                dim = emb_ch,
                heads = 12 if emb_ch%12 == 0 else 8,
                ff_mult = 1,
                attn_dropout=0.0,
                pre_norm = True,
                depth = 1,
                attn_flash = True,
                ff_no_bias = True, 
                rotary_pos_emb=True,
            )
            self.cls_token = nn.Parameter(torch.randn(1, 1, emb_ch))
        elif slice_fusion_type == 'average':
            pass 
        elif slice_fusion_type == "none":
            pass 
        else:
            raise ValueError("Unknown slice_fusion_type")

        self.linear = nn.Linear(emb_ch, out_ch)



    def forward(self, x, output_attentions=False):
        B, *_ = x.shape

        # Mask (Slices with constant padded values)
        x_pad = torch.isclose(x.mean(dim=(-1,-2)), x[:, :, :, 0, 0]) # [B, C, D]
        x_pad = rearrange(x_pad, 'b c d -> b (c d)')

        x = rearrange(x, 'b c d h w -> (b c d) h w')
        x = x[:, None]
        x = x.repeat(1, 3, 1, 1) # Gray to RGB

        # -------------- Backbone --------------
        backbone_out = self.backbone(x, output_attentions=output_attentions)
        x = backbone_out.pooler_output  
        x = rearrange(x, '(b d) e -> b d e', b=B)
 
        # -------------- Slice Fusion --------------
        if self.slice_fusion_type == 'none':
            return x
        elif self.slice_fusion_type == 'transformer':
            cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            pad = torch.concat([x_pad, cls_pad], dim=1)  # [B, D+1]
            x = torch.concat([x, self.cls_token.repeat(B, 1, 1)], dim=1) # [B, 1+D, E]
            if output_attentions:
                x, slice_hiddens = self.slice_fusion(x, mask=~pad, return_hiddens=True) # [B, D+1, E]
            else:
                x = self.slice_fusion(x, mask=~pad) # [B, D+1, L]
        elif self.slice_fusion_type == 'linear':
            x = rearrange(x, 'b d e -> b e d')
            x = self.slice_fusion(x) # ->  [B, E, 1]
            x = rearrange(x, 'b e d -> b d e') #  ->  [B, 1, E]
        elif self.slice_fusion_type == 'average':
            x = x.mean(dim=1, keepdim=True) #  [B, D, E] ->  [B, 1, E]

        # -------------- Logits --------------
        x = self.linear(x[:, -1])
        if output_attentions:
            slice_attn_layers = [
                interm.post_softmax_attn
                for interm in getattr(slice_hiddens, 'attn_intermediates', [])
                if interm is not None and getattr(interm, 'post_softmax_attn', None) is not None
            ]
            return x, backbone_out.attentions, slice_attn_layers
        return x

    def forward_attention(self, x) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, D, _, _ = x.shape
        # Disable fast attention  
        attn_impl = self.backbone.config._attn_implementation
        self.backbone.set_attn_implementation("eager")
        flash_modules = []
        for module in self.slice_fusion.modules():
            if hasattr(module, 'flash'):
                flash_modules.append((module, module.flash))
                module.flash = False

        out, backbone_attn, slice_attn_layers = self.forward(x, output_attentions=True)

        # Restore previous attention implementation
        for module, previous in flash_modules:
            module.flash = previous
        if hasattr(self.backbone, "set_attn_implementation"):
            self.backbone.set_attn_implementation(attn_impl)

        # Process attentions
        slice_attn = torch.stack(slice_attn_layers)[-1]
        slice_attn = slice_attn.mean(dim=1)
        slice_attn = slice_attn[:, -1, :-1]
        slice_attn = slice_attn.view(B, C, D).mean(dim=1)

        plane_attn_layers = [att for att in backbone_attn if att is not None]
        plane_attn = torch.stack(plane_attn_layers)[-1]
        plane_attn = plane_attn.mean(dim=1)
        num_reg_tokens = getattr(self.backbone.config, 'num_register_tokens', 0)
        plane_attn = plane_attn[:, 0, 1 + num_reg_tokens:]
        plane_attn = plane_attn.view(B, C * D, -1)

        # Weight every slice by its slice attention
        plane_attn = plane_attn * slice_attn.unsqueeze(-1)

        num_patches = plane_attn.shape[-1]
        side = int(math.sqrt(num_patches))
        if side * side != num_patches:
            raise RuntimeError("number of patches is not a perfect square")
        plane_attn = plane_attn.reshape(B, C * D, side, side)

        return out, plane_attn, slice_attn

class MST(BasicClassifier):
    # MST - https://arxiv.org/abs/2411.15802 
    def __init__(
            self,
            in_ch=1, 
            out_ch=1, 
            spatial_dims=3,
            backbone_type="dinov3",
            model_size = "s", # 34, 50, ... or 's', 'b', 'l'
            slice_fusion_type = "transformer", # transformer, linear, average, none  
            optimizer_kwargs={'lr':1e-5}, 
            **kwargs
        ):
        super().__init__(in_ch, out_ch, spatial_dims, optimizer_kwargs=optimizer_kwargs, **kwargs)  
        self.mst = _MST(out_ch=out_ch, backbone_type=backbone_type, model_size=model_size, slice_fusion_type=slice_fusion_type)
    
    def forward(self, x):
        return self.mst(x)

    def forward_attention(self, x):
        return self.mst.forward_attention(x)

class MSTRegression(BasicRegression):
    def __init__(
            self,
            in_ch=1, 
            out_ch=1, 
            spatial_dims=3,
            backbone_type="dinov3",
            model_size = "s", # 34, 50, ... or 's', 'b', 'l'
            slice_fusion_type = "transformer", # transformer, linear, average, none  
            optimizer_kwargs={'lr': 1e-5}, 
            **kwargs
        ):
        super().__init__(in_ch, out_ch, spatial_dims, optimizer_kwargs=optimizer_kwargs, **kwargs)  
        self.mst = _MST(out_ch=out_ch, backbone_type=backbone_type, model_size=model_size, slice_fusion_type=slice_fusion_type)
    
    def forward(self, x):
        return self.mst(x)

    def forward_attention(self, x):
        return self.mst.forward_attention(x)
