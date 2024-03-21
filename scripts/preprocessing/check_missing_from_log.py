import pandas as pd

# Load the Excel file to get the list of patient IDs
excel_file_path = '/home/jeff/Athens_data/ODELIA_Paper.xlsx'
df_patient_ids = pd.read_excel(excel_file_path, dtype=str)

# Assuming the patient IDs are in the first column, extract them into a list
patient_ids_from_excel = df_patient_ids.iloc[:, 0].apply(lambda x: x.zfill(14)).tolist()

# Assuming you have a text file with folder names or patient IDs, read it
with open('/home/jeff/PycharmProjects/odelia_breast_mri/scripts/preprocessing/id_txt', 'r') as file:
    lines = file.readlines()

# Extract patient IDs from the document. Here's a generic way to extract assuming they are the last part of a path
patient_ids_from_document = [line.strip() for line in lines]
print(patient_ids_from_document)

# Find which patient IDs from the document are not in the Excel file
missing_patient_ids = set(patient_ids_from_excel) - set(patient_ids_from_document)
print(len(patient_ids_from_document), len(patient_ids_from_excel))
print("Missing patient IDs:", missing_patient_ids)
