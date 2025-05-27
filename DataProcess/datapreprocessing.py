import pandas as pd
from sklearn.model_selection import train_test_split

def clean_and_prepare_data(input_file):
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

    df = pd.read_csv(input_file)

    df = df.drop(columns=cols_to_drop, errors="ignore")
    df = df.dropna(subset=required_cols)
    df = df[df["Overall Survival (Months)"] != 0]

    # Stage encoding
    early_stage = ["STAGE I", "STAGE IA", "STAGE IB"]
    print(df["Neoplasm Disease Stage American Joint Committee on Cancer Code"].unique())
    df["Neoplasm Disease Stage American Joint Committee on Cancer Code"] = df[
        "Neoplasm Disease Stage American Joint Committee on Cancer Code"
    ].apply(lambda x: 0 if str(x).strip().upper() in early_stage else 1)

    df["Overall Survival Status"] = df["Overall Survival Status"].map({"0:LIVING": 0, "1:DECEASED": 1}).astype(int)
    df["Sex"] = df["Sex"].map({"Male": 1, "Female": 0}).astype(int)
    df["TCGA PanCanAtlas Cancer Type Acronym"] = df["TCGA PanCanAtlas Cancer Type Acronym"].map({"LUAD": 0, "LUSC": 1}).astype(int)

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
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, shuffle=True)

    # Save files
    df.to_csv("TCGA-LUNG_all.csv", index=False)
    train_df.to_csv("TCGA-LUNG_train.csv", index=False)
    test_df.to_csv("TCGA-LUNG_test.csv", index=False)

    print("Data processing complete.")
    print("Saved: TCGA-LUNG_all.csv (full cleaned dataset)")
    print("Saved: TCGA-LUNG_train.csv (70% training set)")
    print("Saved: TCGA-LUNG_test.csv (30% test set)")
    return df, train_df, test_df

def compute_statistics(df, name):
    stage_pct = df['Stage'].value_counts(normalize=True) * 100
    sex_pct = df['Sex'].value_counts(normalize=True) * 100
    subtype_pct = df['Subtype'].value_counts(normalize=True) * 100
    age_avg = df['Age'].mean()
    total_samples = len(df)

    return {
        "Dataset": name,
        "Total Samples": total_samples,
        "Stage 0 (%)": stage_pct.get(0, 0),
        "Stage 1 (%)": stage_pct.get(1, 0),
        "Sex 0 (Female) (%)": sex_pct.get(0, 0),
        "Sex 1 (Male) (%)": sex_pct.get(1, 0),
        "Subtype 0 (LUAD) (%)": subtype_pct.get(0, 0),
        "Subtype 1 (LUSC) (%)": subtype_pct.get(1, 0),
        "Average Age": age_avg
    }
