import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 777

def clean_and_prepare_data(input_file: str, file_path: str, split_size: float, split_val_test: bool, scaling: bool):
    # Columns to drop
    cols_to_drop = [
        "Patient ID", "American Joint Committee on Cancer Publication Version Type",
        "Aneuploidy Score", "Buffa Hypoxia Score", "Cancer Type", "Cancer Type Detailed",
        "Last Communication Contact from Initial Pathologic Diagnosis Date",
        "Birth from Initial Pathologic Diagnosis Date",
        "Last Alive Less Initial Pathologic Diagnosis Date Calculated Day Value",
        "Disease Free (Months)", "Disease Free Status", "Months of disease-specific survival",
        "Disease-specific Survival status", "Ethnicity Category", "Form completion date",
        "Fraction Genome Altered", "Neoplasm Histologic Grade",
        "Neoadjuvant Therapy Type Administered Prior To Resection Text",
        "ICD-10 Classification",
        "International Classification of Diseases for Oncology, Third Edition ICD-O-3 Histology Code",
        "International Classification of Diseases for Oncology, Third Edition ICD-O-3 Site Code",
        "Informed consent verified", "In PanCan Pathway Analysis", "MSI MANTIS Score",
        "MSIsensor Score", "Mutation Count", "New Neoplasm Event Post Initial Therapy Indicator",
        "Oncotree Code", "Other Patient ID",
        "American Joint Committee on Cancer Metastasis Stage Code",
        "Neoplasm Disease Lymph Node Stage American Joint Committee on Cancer Code",
        "American Joint Committee on Cancer Tumor Stage Code",
        "Person Neoplasm Cancer Status", "Progress Free Survival (Months)",
        "Progression Free Status", "Primary Lymph Node Presentation Assessment",
        "Prior Diagnosis", "Race Category", "Radiation Therapy", "Ragnum Hypoxia Score",
        "Number of Samples Per Patient", "Sample Type", "Somatic Status", "Subtype",
        "Tissue Prospective Collection Indicator", "Tissue Retrospective Collection Indicator",
        "Tissue Source Site", "Tissue Source Site Code", "TMB (nonsynonymous)",
        "Tumor Disease Anatomic Site", "Tumor Type", "Patient Weight",
        "Winter Hypoxia Score", "miRNA_ID"
    ]

    # Required columns
    required_cols = [
        "Sample ID", "Diagnosis Age", "Neoplasm Disease Stage American Joint Committee on Cancer Code",
        "Overall Survival (Months)", "Overall Survival Status", "Sex",
        "TCGA PanCanAtlas Cancer Type Acronym"
    ]

    df = pd.read_csv(file_path + input_file)

    df = df.drop(columns=cols_to_drop, errors="ignore")
    df = df.dropna(subset=required_cols)
    df = df[df["Overall Survival (Months)"] != 0]

    # Stage encoding
    early_stage = ["STAGE I", "STAGE IA", "STAGE IB"]
    #show stage types
    #print(df["Neoplasm Disease Stage American Joint Committee on Cancer Code"].unique())
    df["Neoplasm Disease Stage American Joint Committee on Cancer Code"] = df[
        "Neoplasm Disease Stage American Joint Committee on Cancer Code"
    ].apply(lambda x: 0 if str(x).strip().upper() in early_stage else 1)

    df["Overall Survival Status"] = df["Overall Survival Status"].map({"0:LIVING": 0, "1:DECEASED": 1}).astype(int)
    df["Sex"] = df["Sex"].map({"Male": 1, "Female": 0}).astype(int)
    df["TCGA PanCanAtlas Cancer Type Acronym"] = df["TCGA PanCanAtlas Cancer Type Acronym"].map({"LUAD": 0, "LUSC": 1}).astype(int)

    #scaling
    if scaling:
        #age and sex are now features
        required_cols.remove("Diagnosis Age")
        required_cols.remove("Sex")

        required_df = df[required_cols]
        features_df = df.drop(columns=required_cols, errors="ignore")
        scaler = StandardScaler()
        features_df = pd.DataFrame(scaler.fit_transform(features_df), columns=features_df.columns, index=features_df.index)

        df = pd.concat([required_df, features_df], axis=1)

    df = df.rename(columns={
        "Sample ID": "ID",
        "Diagnosis Age": "Age",
        "Neoplasm Disease Stage American Joint Committee on Cancer Code": "Stage",
        "TCGA PanCanAtlas Cancer Type Acronym": "Subtype"
    })

    # Reorder columns
    all_columns = df.columns.tolist()
    mirna_cols = [col for col in all_columns if col.startswith("hsa-")]
    new_order = ["ID"] + mirna_cols + ["Stage", "Overall Survival (Months)", "Overall Survival Status", "Age", "Sex", "Subtype"]
    df = df[new_order]

    # Train-test split
    train_df, validate_df = train_test_split(df, test_size=split_size, stratify=df["Stage"], random_state=RANDOM_STATE, shuffle=True)
    if split_val_test: 
        validate_df, test_df =  train_test_split(validate_df, test_size=0.5, stratify=validate_df["Stage"], random_state=RANDOM_STATE, shuffle=True)

    # Save files
    df.to_csv(file_path + "TCGA-LUNG_all.csv", index=False)
    train_df.to_csv(file_path + "TCGA-LUNG_train.csv", index=False)
    validate_df.to_csv(file_path + "TCGA-LUNG_validate.csv", index=False)
    if split_val_test:
        test_df.to_csv(file_path + "TCGA-LUNG_test.csv", index=False)

    print("Data processing complete.")
    print("Saved: TCGA-LUNG_all.csv (full cleaned dataset)")
    print(f"Saved: TCGA-LUNG_train.csv ({100 * (1 - split_size)}% training set)")
    if split_val_test:
        print(f"Saved: TCGA-LUNG_validate.csv ({100 * (split_size / 2)}% test set)")
        print(f"Saved: TCGA-LUNG_test.csv ({100 * (split_size / 2)}% test set)")
    else:
        print(f"Saved: TCGA-LUNG_validate.csv ({100 * (split_size)}% test set)")
        
    return df, train_df, validate_df, test_df if split_val_test else None

def compute_statistics(df, name):
    #age and sex become features for prediction. cannot compute statistic after scaling.
    if type(df) != pd.DataFrame:
        return {
            "Dataset": name,
            "Total Samples": 0,
            "Stage 0 (%)": 0,
            "Stage 1 (%)": 0,
            "Subtype 0 (LUAD) (%)": 0,
            "Subtype 1 (LUSC) (%)": 0,
        }
    
    stage_pct = df['Stage'].value_counts(normalize=True) * 100
    subtype_pct = df['Subtype'].value_counts(normalize=True) * 100
    total_samples = len(df)

    return {
        "Dataset": name,
        "Total Samples": total_samples,
        "Stage 0 (%)": stage_pct.get(0, 0),
        "Stage 1 (%)": stage_pct.get(1, 0),
        "Subtype 0 (LUAD) (%)": subtype_pct.get(0, 0),
        "Subtype 1 (LUSC) (%)": subtype_pct.get(1, 0),
    }
