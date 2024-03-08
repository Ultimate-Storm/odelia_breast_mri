from pathlib import Path
import pydicom
import logging
import sys
import re

from tqdm import tqdm
import SimpleITK as sitk
import numpy as np

# Setting
path_root = Path('/home/jeff/Desktop/data_combridge/dataset')
path_root_out = Path("/home/jeff/Desktop/data_combridge/nifty")




path_root_out.mkdir(parents=True, exist_ok=True)

# Logging
path_log_file = path_root/'preprocessing.log'
logger = logging.getLogger(__name__)
s_handler = logging.StreamHandler(sys.stdout)
f_handler = logging.FileHandler(path_log_file, 'w')
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[s_handler, f_handler])

reader = sitk.ImageSeriesReader()
# get only dir of the path_root
dir_list = [x for x in path_root.iterdir() if x.is_dir()]
print(dir_list)
for case_i, path_dir in enumerate(tqdm(sorted(dir_list))):
    # Only proceed if path is pointing to a directory
    if not path_dir.is_dir():
        continue

    # read the case_id from the folder name, folder name is ODELIA_DEMO_01_01, ODELIA_DEMO_02_01
    case_id = path_dir.name.split('_')[2]

    print(case_id)
    logger.debug(f"Case ID: {case_id}, Number {case_i}")

    # Create output folder
    path_out_dir = path_root_out/case_id
    path_out_dir.mkdir(exist_ok=True)

    # Get all DICOM files in the directory
    dicom_files = sorted(path_dir.glob('*.dcm'))

    # Separate the DICOM files into pre-contrast and post-contrast
    dicom_files_pre = [str(f) for f in dicom_files if 'Pre' in f.name]
    dicom_files_post = [str(f) for f in dicom_files if 'Post' in f.name]

    # Initialize an empty list to store the subtraction images
    sub_images = []
    pre_images = []

    # For each pair of pre and post-contrast DICOMs, compute the subtraction image
    for pre_file, post_file in zip(dicom_files_pre, dicom_files_post):
        reader.SetFileNames([pre_file])
        img_nii_pre = reader.Execute()

        reader.SetFileNames([post_file])
        img_nii_post = reader.Execute()

        # Compute subtraction image
        logger.debug(f"Compute and write sub to disk")
        dyn0 = sitk.GetArrayFromImage(img_nii_pre)
        dyn1 = sitk.GetArrayFromImage(img_nii_post)
        sub = dyn1 - dyn0
        sub = sub - sub.min()  # Note: negative values causes overflow when using uint
        sub = sub.astype(np.uint16)

        # Append the subtraction image to the list
        sub_images.append(sub)
        pre_images.append(dyn0)

    # Check if sub_images is not empty
    if sub_images:
        # Convert the list of subtraction images into a 3D numpy array
        sub_3d = np.stack(sub_images)
        print(sub_3d.shape)# (96, 1, 512, 512)
        # stack the images along the first axis
        sub_3d = np.squeeze(sub_3d, axis=1)
        print(sub_3d.shape)
        # Save the 3D array as a NIfTI file
        sub_nii = sitk.GetImageFromArray(sub_3d)
        sub_nii.SetSpacing(img_nii_pre.GetSpacing())
        sub_nii.SetOrigin(img_nii_pre.GetOrigin())
        sub_nii.SetDirection(img_nii_pre.GetDirection())
        sitk.WriteImage(sub_nii, str(path_out_dir/'sub.nii.gz'))

    # Check if pre_images is not empty
    if pre_images:
        # Convert the list of pre images into a 3D numpy array
        pre_3d = np.stack(pre_images)
        print(pre_3d.shape)# (96, 1, 512, 512)
        # stack the images along the first axis
        pre_3d = np.squeeze(pre_3d, axis=1)
        print(pre_3d.shape)
        # Save the 3D array as a NIfTI file
        pre_nii = sitk.GetImageFromArray(pre_3d)
        pre_nii.SetSpacing(img_nii_pre.GetSpacing())
        pre_nii.SetOrigin(img_nii_pre.GetOrigin())
        pre_nii.SetDirection(img_nii_pre.GetDirection())
        sitk.WriteImage(pre_nii, str(path_out_dir/'pre.nii.gz'))
