import os
import pydicom
from collections import defaultdict


def get_sequence_names(directory):
    sequence_names = defaultdict(int)
    folder_name = None

    for root, dirs, files in os.walk(directory):
        for file in files:
            if not file.endswith('.DCM'):  # Skip non-DICOM files
                continue
            try:
                filepath = os.path.join(root, file)
                dicom_file = pydicom.dcmread(filepath)
                sequence_name = dicom_file.get((0x0008, 0x103E), None)  # Sequence Name
                if sequence_name is not None:
                    sequence_names[str(sequence_name.value)] += 1
                    if str(sequence_name.value) == 't1_fl3d_spair_tra_p3_caipi_dynaVIEWS_1+4_rec_SUB':
                        folder_name = root
                    #print(f"Folder: {root}, Sequence Name: {sequence_name.value}")
            except Exception as e:
                print(f"Error reading {file}: {e}")
                continue
    return sequence_names, folder_name

if __name__ == "__main__":
    directory = "/home/jeff/Downloads/OneDrive_1_3-8-2024/ANON_03b27b8125fc4b86b91826d4dc76ffe0"
    sequence_names, folder_name = get_sequence_names(directory)
    if sequence_names:
        print("Sequence Names found in DICOM files:")
        #for sequence_name, count in sequence_names.items():
           # print(f"{sequence_name}: {count} files")
    else:
        print("No DICOM files with sequence names found.")
    if folder_name:
        print(f"Folder containing 't1_fl3d_spair_tra_p3_caipi_dynaVIEWS_1+4_rec_SUB': {folder_name}")
    else:
        print("No folder found containing 't1_fl3d_spair_tra_p3_caipi_dynaVIEWS_1+4_rec_SUB'")