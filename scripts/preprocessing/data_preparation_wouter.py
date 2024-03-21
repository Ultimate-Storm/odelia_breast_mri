from pathlib import Path
import SimpleITK as sitk
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

path_dir = Path("/media/jeff/TOSHIBA EXT/ODELIA_ALL_data_processed_woulter/Extravted_NII/new_data_pre_post_no_HR")
path_out_dir = Path("/home/jeff/wouter")  # Assuming you want to output in the same directory
path_out_dir.mkdir(exist_ok=True)

# Gather files and organize by patient ID
file_dict = {}
files = list(path_dir.glob('*.nii.gz'))
for file in files:
    patient_id, time_point = file.stem.split('_')
    time_point = time_point.split('.')[0]  # Extract last character
    #print(patient_id, time_point)
    if time_point in ['0', '4']:  # Only consider pre (0) and post (4) time points
        if patient_id not in file_dict:
            file_dict[patient_id] = {}
        file_dict[patient_id][time_point] = file
print(file_dict)
# Process each patient
for patient_id, time_points in file_dict.items():
    try:
        pre_file = time_points.get('0')
        post_file = time_points.get('4')
        print(pre_file, post_file)
        if not pre_file or not post_file:
            logging.warning(f"Missing pre or post file for patient {patient_id}. Skipping...")
            continue

        # Compute subtraction image
        pre_img = sitk.ReadImage(str(pre_file), sitk.sitkInt16)
        post_img = sitk.ReadImage(str(post_file), sitk.sitkInt16)

        pre_arr = sitk.GetArrayFromImage(pre_img)
        post_arr = sitk.GetArrayFromImage(post_img)

        sub_arr = post_arr - pre_arr
        sub_arr -= sub_arr.min()  # Shift to make all values positive

        sub_img = sitk.GetImageFromArray(sub_arr.astype(np.uint16))
        sub_img.CopyInformation(pre_img)

        output_filename = path_out_dir / f"{patient_id}_sub.nii.gz"
        sitk.WriteImage(sub_img, str(output_filename))
        logging.info(f"Processed and saved subtraction image for patient {patient_id} as {output_filename.name}")
    except Exception as e:
        logging.error(f"Error processing patient {patient_id}: {e}")

