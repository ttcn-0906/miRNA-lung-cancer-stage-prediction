TCGA-LUNG Data Preprocessing and Statistics Pipeline
====================================================

📁 Files Included:
------------------
1. main.py
   - Entry point for running the full data processing and statistics pipeline.
2. datapreprocessing.py
   - Contains functions for:
     • Cleaning the TCGA-LUNG dataset
     • Label encoding of clinical fields
     • Train-test splitting (70:30)
     • Basic statistics generation
3. TCGA-LUNG_raw_data.csv
   - The original input dataset (should be placed in the same directory).
4. Output Files (Generated After Running main.py):
   - TCGA-LUNG_all.csv   → Cleaned and processed full dataset
   - TCGA-LUNG_train.csv → 70% training data
   - TCGA-LUNG_test.csv  → 30% test data
   - TCGA-LUNG_stats_summary.csv → Summary statistics

🛠 How to Run:
--------------
1. Make sure Python ≥ 3.7 is installed.
2. Install required dependencies:
   pip install pandas scikit-learn
3. Place the original TCGA-LUNG_raw_data.csv in the same folder.
4. Run the script:
   python main.py

📊 What It Does:
----------------
- Removes irrelevant or redundant columns from the original CSV.
- Drops rows with missing key clinical variables or survival time = 0.
- Encodes the following clinical fields:
    • Stage: Early (0) or Late (1)
    • Sex: Female (0), Male (1)
    • Subtype: LUAD (0), LUSC (1)
- Renames key columns (e.g., Diagnosis Age → Age, Sample ID → ID).
- Reorders columns: ID + miRNAs → Stage/OS/Status/Age/Sex/Subtype.
- Splits data into training and test sets (70/30).
- Outputs clean datasets and descriptive statistics (including sample count, average age, and label distributions).
