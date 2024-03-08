from pathlib import Path
import torchio as tio
import torch
from multiprocessing import Pool

from odelia.data.augmentation.augmentations_3d import CropOrPad

def crop_breast_height(image, margin_top=10):
    "Crop height to 256 and try to cover breast based on intensity localization"
    threshold = int(image.data.float().quantile(0.9))
    foreground = image.data>threshold
    fg_rows = foreground[0].sum(axis=(0, 2))
    top = min(max(512-int(torch.argwhere(fg_rows).max()) - margin_top, 0), 256)
    bottom = 256-top
    return  tio.Crop((0,0, bottom, top, 0, 0))(image)

def preprocess(path_img):
    # -------- Settings --------------
    print(path_img)
    target_shape = (512, 512, 32)
    target_spacing = (0.7, 0.7, 3)

    # Read image
    img = tio.ScalarImage(path_img)
    print(img.shape)

    # Preprocess (eg. Crop/Pad)
    transform = tio.Compose([
        tio.Resample(target_spacing),

        CropOrPad(target_shape, padding_mode='mean'),
        tio.ToCanonical(),
    ])
    img = transform(img)

    # Crop bottom and top so that height is 256 and breast is preserved
    img = crop_breast_height(img)

    # Split left and right side
    img_left = tio.Crop((0, 256, 0, 0, 0, 0))(img)
    img_right = tio.Crop((256, 0, 0, 0, 0, 0))(img)

    # Save
    img_left.save(path_img.with_name( '_left.nii.gz'))
    img_right.save(path_img.with_name( '_right.nii.gz'))

if __name__ == "__main__":
    path_img = Path('/home/jeff/Downloads/OneDrive_1_3-8-2024/ANON_594e9d58d1174224a722a103f2e60e71/DIR000/00000023_nifty/image.nii.gz')
    preprocess(path_img)