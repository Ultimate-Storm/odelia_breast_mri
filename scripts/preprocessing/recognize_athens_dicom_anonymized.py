import os
import pydicom
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed


def process_file(filepath):
    try:
        dicom_file = pydicom.dcmread(filepath)
        sequence_name = dicom_file.get((0x0008, 0x103E), None)  # Sequence Name
        if sequence_name is not None:
            return str(sequence_name.value), filepath
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return None, None


def get_sequence_names(directory):
    sequence_names = defaultdict(int)
    folder_names = defaultdict(set)

    with ThreadPoolExecutor() as executor:
        # Create a list of future tasks
        futures = []
        for root, _, files in os.walk(directory):
            for file in files:
                filepath = os.path.join(root, file)
                futures.append(executor.submit(process_file, filepath))

        # Process completed tasks
        for future in as_completed(futures):
            sequence_name, filepath = future.result()
            if sequence_name:
                sequence_names[sequence_name] += 1
                if sequence_name == 't1_fl3d_spair_tra_p3_caipi_dynaVIEWS_1+4_rec_SUB':
                    folder_names[sequence_name].add(os.path.dirname(filepath))

    # Select the folder name if exists
    target_sequence = 't1_fl3d_spair_tra_p3_caipi_dynaVIEWS_1+4_rec_SUB'
    selected_folder = sorted(folder_names[target_sequence])[-1] if target_sequence in folder_names else None

    return sequence_names, selected_folder


if __name__ == "__main__":
    directory = "/home/jeff/extract"
    folder_list = []
    for dir in os.listdir(directory):
        sequence_names, folder_name = get_sequence_names(os.path.join(directory, dir))
        if folder_name:
            print(f"Folder containing 't1_fl3d_spair_tra_p3_caipi_dynaVIEWS_1+4_rec_SUB': {folder_name}")
        else:
            print("No folder found containing 't1_fl3d_spair_tra_p3_caipi_dynaVIEWS_1+4_rec_SUB'")
        folder_list.append(folder_name)
