from pathlib import Path
import SimpleITK as sitk
from recognize_athens_dicom_anonymized import get_sequence_names
import os
from concurrent.futures import ProcessPoolExecutor


def convert_dicom_to_nifty(folder):
    if folder is None:
        return None
    try:
        print(f"Converting DICOM to NIfTI for {folder}")
        path_root = Path(folder)
        path_root_out = Path(f"{path_root}_nifty")
        # Ensure output directory exists
        path_root_out.mkdir(parents=True, exist_ok=True)

        # Setup SimpleITK reader
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(str(path_root))
        reader.SetFileNames(dicom_names)
        image = reader.Execute()

        # Write NIfTI image
        sitk.WriteImage(image, str(path_root_out / 'sub.nii.gz'))
        return str(path_root_out)
    except Exception as e:
        print(f"Error converting DICOM to NIfTI for {folder}: {e}")
        return None


def main(directory):
    folder_list = []
    with ProcessPoolExecutor() as executor:
        # First, get the sequence names and folder names
        for dir in os.listdir(directory):
            sequence_names, folder_name = get_sequence_names(os.path.join(directory, dir))
            if folder_name:
                print(f"Folder containing 't1_fl3d_spair_tra_p3_caipi_dynaVIEWS_1+4_rec_SUB': {folder_name}")
            else:
                print("No folder found containing 't1_fl3d_spair_tra_p3_caipi_dynaVIEWS_1+4_rec_SUB'")
            folder_list.append(folder_name)

        # Then, convert each DICOM folder to NIfTI in parallel
        results = list(executor.map(convert_dicom_to_nifty, folder_list))

        # Optional: Process or print the results
        for result in results:
            if result:
                print(f"Conversion completed: {result}")
            else:
                print("Conversion failed or skipped for a folder.")


if __name__ == "__main__":
    directory = "/home/jeff/Athens_data/"
    main(directory)
