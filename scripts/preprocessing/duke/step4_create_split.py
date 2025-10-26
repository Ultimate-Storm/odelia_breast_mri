from pathlib import Path 
import numpy as np 
import pandas as pd 

from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def create_split(df, uid_col='UID', label_col='Label', group_col='PatientID'):
    df = df.reset_index(drop=True)
    splits = []
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0) # StratifiedGroupKFold
    sgkf2 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    for fold_i, (train_val_idx, test_idx) in enumerate(sgkf.split(df[uid_col], df[label_col], groups=df[group_col])):
        df_split = df.copy()
        df_split['Fold'] = fold_i 
        df_trainval = df_split.iloc[train_val_idx]
        train_idx, val_idx = list(sgkf2.split(df_trainval[uid_col], df_trainval[label_col], groups=df_trainval[group_col]))[0]
        train_idx, val_idx = df_trainval.iloc[train_idx].index, df_trainval.iloc[val_idx].index 
        df_split.loc[train_idx, 'Split'] = 'train' 
        df_split.loc[val_idx, 'Split'] = 'val' 
        df_split.loc[test_idx, 'Split'] = 'test' 
        splits.append(df_split)
    df_splits = pd.concat(splits)
    return df_splits 


def load_annotation(path_root_metadata:Path):
    df = pd.read_excel(path_root_metadata/'Clinical_and_Other_Features.xlsx', header=[0, 1, 2])
    df = df[df[df.columns[38]] != 'NC'] # check if cancer is bilateral=1, unilateral=0 or NC 
    df = df[[df.columns[0], df.columns[19], df.columns[36],  df.columns[38]]] # Only pick relevant columns: Patient ID, Age, Tumor Side, Bilateral
    df.columns = ['PatientID', 'Age', 'Location', 'Bilateral']  # Simplify columns as: Patient ID, Age, Tumor Side, Bilateral
    dfs = []
    for side in ["left", 'right']:
        dfs.append(pd.DataFrame({
            'PatientID': df["PatientID"],
            'UID': df["PatientID"] + f"_{side}",
            'Age': df['Age'].abs(), # turn negative ages to positive
            'Lesion':df[["Location", "Bilateral"]].apply(lambda ds: int((ds[0] == side[0].upper()) | (ds[1]==1)) , axis=1) } ) )
    df = pd.concat(dfs,  ignore_index=True)
    df = df.reset_index(drop=True)
    return df


if __name__ == "__main__":
    path_root = Path('/home/homesOnMaster/gfranzes/Documents/datasets/ODELIA/')
    path_root_dataset = path_root/'DUKE'
    path_root_metadata = path_root_dataset/'metadata_unilateral'


    df = load_annotation(path_root_metadata)

    # Adapt to ODELIA standard
    df['Lesion'] = df['Lesion'].map({0:0, 1:2}) # 0: no lesion, 2: malignant lesion

    # Create annotation dataframe
    df_anno = df[['UID', 'PatientID', 'Age', 'Lesion']]
    df_anno.to_csv(path_root_metadata/'annotation.csv', index=False)


    print("Patients", df['PatientID'].nunique())
    print("Breasts", df['UID'].nunique())
    for lesion_type, count in df['Lesion'].value_counts().sort_index().items():
        print(f"Lesion Type {lesion_type}: {count}")

    df_splits = create_split(df, uid_col='UID', label_col='Lesion', group_col='PatientID')
    df_splits = df_splits[['UID', 'Fold', 'Split']]
    df_splits.to_csv(path_root_metadata/'split.csv', index=False)