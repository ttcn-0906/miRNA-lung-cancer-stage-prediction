import pandas as pd
import os
from datapreprocessing import clean_and_prepare_data, compute_statistics

# Run the pipeline
input_file = "TCGA-LUNG_raw_data.csv"
df, train_df, test_df = clean_and_prepare_data(input_file)

# Compute and print statistics
stats_all = compute_statistics(df, "All Data")
stats_train = compute_statistics(train_df, "Train Data")
stats_test = compute_statistics(test_df, "Test Data")
stats_df = pd.DataFrame([stats_all, stats_train, stats_test])
pd.set_option('display.max_columns', None)
print("\n=== Summary Statistics ===")
print(stats_df)