from pathlib import Path
import SimpleITK as sitk

# Setting
path_root = Path(' /home/jeff/Downloads/OneDrive_1_3-8-2024/ANON_03b27b8125fc4b86b91826d4dc76ffe0/DIR000/00000013')
path_root_out = Path(str(path_root)+"_nifty")
# get all dicoms under path_root
dicom_files = list(path_root.glob('*.DCM'))
path_root_out.mkdir(parents=True, exist_ok=True)
# convert dicom to nifty
reader = sitk.ImageSeriesReader()
dicom_names = reader.GetGDCMSeriesFileNames(str(path_root))
reader.SetFileNames(dicom_names)
image = reader.Execute()
sitk.WriteImage(image, str(path_root_out/'image.nii.gz'))
