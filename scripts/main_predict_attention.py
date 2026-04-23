import argparse
from pathlib import Path
from tqdm import tqdm
import torch 
import numpy as np
import torch.nn.functional as F
import torchio as tio
from torchvision.utils import save_image
from matplotlib.pyplot import get_cmap

from odelia.models import MSTRegression



def minmax_norm(x):
    """Normalizes input to [0, 1] for each batch and channel"""
    return (x - x.min()) / (x.max() - x.min())

def tensor2image(tensor, batch=0):
    """Transform tensor into shape of multiple 2D RGB/gray images. """
    return (tensor if tensor.ndim<5 else torch.swapaxes(tensor[batch], 0, 1).reshape(-1, *tensor.shape[-2:])[:,None])

def tensor_cam2image(tensor, cam, batch=0, alpha=0.5, color_map=get_cmap('jet')):
    """Transform a tensor and a (grad) cam into multiple 2D RGB images."""
    img = tensor2image(tensor, batch) #  -> [B, C, H, W]
    img = torch.cat([img for _ in range(3)], dim=1) if img.shape[1]!=3 else img # Ensure RGB  [B, 3, H, W] 
    cam_img = tensor2image(cam, batch) #  -> [B, 1, H, W]
    cam_img = cam_img[:,0].cpu().numpy() # -> [B, H, W]
    cam_img = torch.tensor(color_map(cam_img)) # -> [B, H, W, 4], color_map expects input to be [0.0, 1.0]
    cam_img = torch.moveaxis(cam_img, -1, 1)[:, :3] # -> [B, 3, H, W]
    overlay = (1-alpha)*img + alpha*cam_img
    return overlay



def crop_breast_height(image, margin_top=10) -> tio.Crop:
    """Crop height to 256 and try to cover breast based on intensity localization"""
    threshold = int(np.quantile(image.data.float(), 0.9))
    foreground = image.data>threshold
    fg_rows = foreground[0].sum(axis=(0, 2))
    top = min(max(512-int(torch.argwhere(fg_rows).max()) - margin_top, 0), 256)
    bottom = 256-top
    return  tio.Crop((0,0, bottom, top, 0, 0))


def get_bilateral_transform(img:tio.ScalarImage, ref_img=None, target_spacing = (0.7, 0.7, 3), target_shape = (512, 512, 32)):
    # -------- Settings --------------
    ref_img = img if ref_img is None else ref_img
    
    # Spacing 
    ref_img = tio.ToCanonical()(ref_img)
    ref_img = tio.Resample(target_spacing)(ref_img)
    resample = tio.Resample(ref_img)

    # Crop 
    ref_img = tio.CropOrPad(target_shape, padding_mode='minimum')(ref_img)
    crop_height = crop_breast_height(ref_img)     

    # Process input image
    trans = tio.Compose([
        resample,
        tio.CropOrPad(target_shape, padding_mode='minimum'),
        crop_height,
    ])

    trans_inv = tio.Compose([
        crop_height.inverse(),
        tio.CropOrPad(img.spatial_shape, padding_mode='minimum'),
        tio.Resample(img),
    ])
    return trans(img), trans_inv

def get_unilateral_transform(img: tio.ScalarImage, target_shape=(224, 224, 32)):
    transform = tio.Compose([
        tio.Flip((1,0)), 
        tio.CropOrPad(target_shape),
        tio.ZNormalization(masking_method=lambda x:(x>x.min()) & (x<x.max())), 
    ])
    inv_transform = tio.Compose([
        tio.CropOrPad(img.spatial_shape),
        tio.Flip((1,0)), 
    ])
    return transform(img), inv_transform


def run_prediction(img: tio.ScalarImage, model: MSTRegression):
    img_bil, bil_trans_rev = get_bilateral_transform(img)
    split_side = {
        'right': tio.Crop((256, 0, 0, 0, 0, 0)),
        'left': tio.Crop((0, 256, 0, 0, 0, 0)),
    }

    weights, probs = {}, {}
    for side, crop in split_side.items():
        img_side = crop(img_bil)
        img_side, uni_trans_inv = get_unilateral_transform(img_side)
        img_side = img_side.data.swapaxes(1,-1)
        img_side = img_side.unsqueeze(0)  # Add batch dim -> [1, C, H, W, D]

        with torch.no_grad():
            device = next(model.parameters()).device
            logits, weight, weight_slice = model.forward_attention(img_side.to(device))

        weight = F.interpolate(weight.unsqueeze(1), size=img_side.shape[2:], mode='trilinear', align_corners=False).cpu()
        # pred_prob = model.logits2probabilities(logits).cpu()
        pred_prob = F.softmax(logits, dim=-1).cpu()
        probs[side] = pred_prob.squeeze(0)

        weight = weight.squeeze(0).swapaxes(1,-1)  # ->[C, W, H, D]
        weight = uni_trans_inv(weight)
        weights[side] = weight

    weight = torch.concat([weights['left'], weights['right']], dim=1) #  C, W, H, D
    weight = tio.ScalarImage(tensor=weight, affine=img_bil.affine)  
    weight = bil_trans_rev(weight)
    weight.set_data(minmax_norm(weight.data))  
    return probs, weight

if __name__ == "__main__":
    #------------ Get Arguments ----------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_run', default='runs/ODELIA/MST_ordinal_unilateral_2025_10_25_140313_fold0/epoch=38-step=3978.ckpt', type=str)
    parser.add_argument('--path_img', default='/home/homesOnMaster/gfranzes/Documents/datasets/ODELIA/UKA/data/UKA_2/Sub_1.nii.gz', type=str)
    args = parser.parse_args()


    #------------ Settings/Defaults ----------------
    path_out_dir = Path().cwd()/'results/test_attention'
    path_out_dir.mkdir(parents=True, exist_ok=True)


    # ------------ Load Data ----------------
    path_img = Path(args.path_img)
    img = tio.ScalarImage(path_img)


    # ------------ Initialize Model ------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path_run = Path(args.path_run) 
    model = MSTRegression.load_from_checkpoint(path_run)
    model.to(device)
    model.eval()


    # ------------ Predict ----------------
    probs, weight = run_prediction(img, model)

    img.save(path_out_dir/f"input.nii.gz")
    weight.save(path_out_dir/f"attention.nii.gz")
    weight = weight.data.swapaxes(1,-1).unsqueeze(0)  # C, D, H, W
    img = img.data.swapaxes(1,-1).unsqueeze(0)  # C, D, H, W
    save_image(tensor_cam2image(minmax_norm(img), minmax_norm(weight), alpha=0.5), 
            path_out_dir/f"overlay.png", normalize=False)
    
    for side in ['left', 'right']:
        print(f"{side} breast predicted probabilities: {probs[side]}")
    
