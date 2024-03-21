from pathlib import Path
import torchio as tio
import torch
from multiprocessing import Pool
from odelia.data.augmentation.augmentations_3d import CropOrPad

def crop_breast_height(image, margin_top=10):
    """Crop height to 256 and try to cover breast based on intensity localization."""
    threshold = int(image.data.float().quantile(0.9))
    foreground = image.data > threshold
    fg_rows = foreground[0].sum(axis=(0, 2))
    top = min(max(512 - int(torch.argwhere(fg_rows).max()) - margin_top, 0), 256)
    bottom = 256 - top
    return tio.Crop((0, 0, bottom, top, 0, 0))(image)

def preprocess(path_img):
    # -------- Settings --------------
    print(path_img)
    target_shape = (512, 512, 32)
    target_spacing = (0.7, 0.7, 3)

    # Read image
    img = tio.ScalarImage(path_img)
    print(img.shape)

    # Preprocess (e.g., Crop/Pad)
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

    # Construct new save path
    parent_dir_name = path_img.parents[1].name  # Adjust based on your directory structure
    save_path_right = Path(f"/home/jeff/preprocessed_re/{path_img.stem.split('.')[0].split('_')[0]}_right")
    save_path_right.mkdir(parents=True, exist_ok=True)

    save_path_left = Path(f"/home/jeff/preprocessed_re/{path_img.stem.split('.')[0].split('_')[0]}_left")
    save_path_left.mkdir(parents=True, exist_ok=True)

    # Save
    img_left.save(save_path_left / "sub.nii.gz")
    img_right.save(save_path_right / "sub.nii.gz")

if __name__ == "__main__":
    # Get all the .nii.gz files recursively in the directory
    path_img = Path('/home/jeff/wouter/')
    nifti_files = list(path_img.rglob('*.nii.gz'))
    print(nifti_files)

    # Preprocess each file and save with the name of its grandparent directory as part of the path
    with Pool(4) as p:
        p.map(preprocess, nifti_files)
