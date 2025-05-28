import pandas as pd
import os
try:
    from .datapreprocessing import clean_and_prepare_data, compute_statistics
except ImportError:
    from datapreprocessing import clean_and_prepare_data, compute_statistics

def start(input_file: str = "TCGA-LUNG_raw_data.csv", file_path: str = "./", validate_size: float = 0.3, split_test: bool = False):
    # Run the pipeline
    df, train_df, validate_df, test_df = clean_and_prepare_data(input_file, file_path, validate_size, split_test)

    # Compute and print statistics
    stats_all = compute_statistics(df, "All Data")
    stats_train = compute_statistics(train_df, "Train Data")
    stats_validate = compute_statistics(validate_df, "Validate Data")
    if split_test:
        stats_test = compute_statistics(test_df, "Test Data")
        stats_df = pd.DataFrame([stats_all, stats_train, stats_validate, stats_test])
    else:
        stats_df = pd.DataFrame([stats_all, stats_train, stats_validate])
    pd.set_option('display.max_columns', None)
    print("\n=== Summary Statistics ===")
    print(stats_df)
    print("==========================\n")

if __name__ == '__main__':
    start()