import torch 
from odelia.models import MST, MSTRegression

input = torch.randn((1,2,32, 224,224))
input[0,0, 0] = 0.0  # Simulate empty slice
# model = MST(in_ch=1, out_ch=2, spatial_dims=3)
model = MSTRegression(in_ch=1, backbone_type="dinov3", out_ch=2+3, spatial_dims=3, loss_kwargs={"class_labels_num": [2,3]})


pred = model(input)
print(pred.shape)
pred, plane_attn, slice_attn = model.forward_attention(input)
print(plane_attn.shape)
print(slice_attn.shape)

