import pandas as pd

# Replace the following path with the actual path to your Excel file
excel_file_path = '/opt/hpe/odelia_breast_mri/dataset/Cancer_RL_labels_20231219_example (1).xlsx'
# Replace the following path with the desired output path for the CSV file
csv_output_path = '/opt/hpe/odelia_breast_mri/dataset/cambridge.csv'

# Read the original data from the Excel file
original_df = pd.read_excel(excel_file_path)

# Now, let's transform the DataFrame to the desired format
transformed_data = []

for index, row in original_df.iterrows():
    transformed_data.append([f"{row['ODELIA_ID']}_left", row['Left']])
    transformed_data.append([f"{row['ODELIA_ID']}_right", row['Right']])

# Creating a DataFrame from the transformed data
transformed_df = pd.DataFrame(transformed_data, columns=['PATIENT', 'Malign'])

# Save the transformed DataFrame to a CSV file
transformed_df.to_csv(csv_output_path, index=False, header=True)
