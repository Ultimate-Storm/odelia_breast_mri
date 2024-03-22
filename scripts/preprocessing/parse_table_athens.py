import pandas as pd

def transform_table(input_excel_path, output_csv_path):
    # Read the excel file
    df = pd.read_excel(input_excel_path , dtype=str)

    # Initialize an empty list to store the transformed data
    transformed_data = []

    for _, row in df.iterrows():
        patient_id = row['MHA Exam ID']
        print(patient_id)
        cancer = 1 if row['Cancer'] == 'Y' else 0
        side = row['Side']

        # For sides L, R, and B (both), transform the data accordingly
        if side in ['L', 'R']:
            # transform L to left and R to right
            sidename = 'left' if side == 'L' else 'right'
            transformed_data.append([f'{patient_id}_{sidename}', cancer])
            # Add the opposite side with 0 cancer if side is L or R
            opposite_side = 'left' if side == 'R' else 'right'
            transformed_data.append([f'{patient_id}_{opposite_side}', 0])
        elif side == 'B':
            # If side is B (both), add both sides with cancer
            transformed_data.append([f'{patient_id}_left', cancer])
            transformed_data.append([f'{patient_id}_right', cancer])
        else:
            # If no side information is provided, assume both sides need to be listed with 0 cancer
            transformed_data.append([f'{patient_id}_left', 0])
            transformed_data.append([f'{patient_id}_right', 0])

    # Convert the list to a DataFrame
    transformed_df = pd.DataFrame(transformed_data, columns=['PATIENT', 'Malign'])

    # Save the transformed DataFrame to a CSV file
    transformed_df.to_csv(output_csv_path, index=False)

# Example usage:
input_excel_path = '/home/jeff/Athens_data/ODELIA_Paper.xlsx'
output_csv_path = '/home/jeff/Downloads/athens_datasheet.csv'
transform_table(input_excel_path, output_csv_path)
