import pandas as pd
import os

# Path to your Excel file
xlsx_file_path = '/home/jeff/Odelia_Paper_149_anon_exams/ODELIA_Paper.xlsx'
# Path to the folder containing the patientID.zip files
folder_path = '/home/jeff/Odelia_Paper_149_anon_exams/'

# Read the first column from the Excel file
df_patient_ids = pd.read_excel(xlsx_file_path, usecols=[0], dtype=str)
print(df_patient_ids)
# the  0 at the beginning was removed, fix this
patient_ids_from_xlsx = df_patient_ids[df_patient_ids.columns[0]].astype(str).tolist()
print("Patient IDs from Excel file:", patient_ids_from_xlsx)

# List all .zip files in the specified folder
available_zip_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.zip')]

# Extract patient IDs from the filenames (assuming filename format is patientID.zip)
patient_ids_from_zip_files = [f.split('.')[0] for f in available_zip_files]

# Find which patient IDs from the Excel file are missing in the .zip files
missing_patient_ids = [pid for pid in patient_ids_from_xlsx if pid not in patient_ids_from_zip_files]

# Print the missing patient IDs
print("Missing patient IDs:", missing_patient_ids)

# get the missing patient IDs from the  zip files, which is missing in the excel file
missing_patient_ids = [pid for pid in patient_ids_from_zip_files if pid not in patient_ids_from_xlsx]
print("Missing patient IDs:", missing_patient_ids)

