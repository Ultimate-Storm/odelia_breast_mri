import os
import pydicom
from collections import defaultdict

sub_folders = []
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
                        sub_folders.append(folder_name)
                    #print(f"Folder: {root}, Sequence Name: {sequence_name.value}")
            except Exception as e:
                print(f"Error reading {file}: {e}")
                continue
    return sequence_names, sub_folders

if __name__ == "__main__":
    directory = "/home/jeff/Downloads/OneDrive_1_3-8-2024/ANON_594e9d58d1174224a722a103f2e60e71//"
    sequence_names, folder_name = get_sequence_names(directory)
    if sequence_names:
        print("Sequence Names found in DICOM files:")
        #for sequence_name, count in sequence_names.items():
           # print(f"{sequence_name}: {count} files")
    else:
        print("No DICOM files with sequence names found.")

    # sub_folders remove duplicates
    sub_folders = list(set(sub_folders))
    #print(sub_folders)
    print(len(sub_folders))
    # print the number of DICOM files in each subfolder
    for sub_folder in sub_folders:
        print(f"Subfolder: {sub_folder}")
        # how many DICOM files in each subfolder ends with .DCM
        print(len([name for name in os.listdir(sub_folder) if name.endswith('.DCM')]))
    if folder_name:
        print(f"Folder containing 't1_fl3d_spair_tra_p3_caipi_dynaVIEWS_1+4_rec_SUB': {sub_folders}")
    else:
        print("No folder found containing 't1_fl3d_spair_tra_p3_caipi_dynaVIEWS_1+4_rec_SUB'")


    # print the meta data of the /home/jeff/Downloads/OneDrive_1_3-8-2024/ANON_594e9d58d1174224a722a103f2e60e71//DIR000/00000012
    # folder
    for root, dirs, files in os.walk(sub_folders[0]):
        for file in files:
            if not file.endswith('.DCM'):  # Skip non-DICOM files
                continue
            try:
                filepath = os.path.join(root, file)
                dicom_file = pydicom.dcmread(filepath)
                print(f"Folder: {root}, File: {file}, Meta Data: {dicom_file}")
                break
            except Exception as e:
                print(f"Error reading {file}: {e}")
                continue

    for root, dirs, files in os.walk(sub_folders[2]):
        for file in files:
            if not file.endswith('.DCM'):  # Skip non-DICOM files
                continue
            try:
                filepath = os.path.join(root, file)
                dicom_file = pydicom.dcmread(filepath)
                print(f"Folder: {root}, File: {file}, Meta Data: {dicom_file}")
                break
            except Exception as e:
                print(f"Error reading {file}: {e}")
                continue