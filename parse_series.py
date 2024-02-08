import os
import pydicom

def find_dicom_files(directory):
    """Recursively find DICOM files in a directory."""
    dicom_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.dcm'):
                dicom_files.append(os.path.join(root, file))
    print(dicom_files)
    return dicom_files

def print_metadata(dicom_files):
    """Print metadata for each DICOM file."""
    for dicom_file in dicom_files:
        try:
            ds = pydicom.dcmread(dicom_file, stop_before_pixels=True)
            print(f"Metadata for {dicom_file}:")
            for elem in ds:
                try:
                    print(f"  {elem.keyword}: {elem.value}")
                except Exception as keyword_exception:
                    print(f"  Error retrieving keyword for {elem.tag}: {keyword_exception}")
            print("\n---\n")
        except Exception as e:
            print(f"Error reading {dicom_file}: {e}")

def main(directory):
    """Main function to process DICOM files."""
    dicom_files = find_dicom_files(directory)
    print_metadata(dicom_files)

# Adjust the directory path to match your specific setup
directory_path = '/opt/hpe/odelia_breast_mri/original_data/'
main(directory_path)
