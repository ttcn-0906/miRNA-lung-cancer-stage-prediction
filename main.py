from DataProcess import process
from Deep import FCNN_training
from Baseline import logisticregression

#data process
raw_file_path = "DataProcess/"
raw_file_name = "TCGA-LUNG_raw_data.csv"
split_val_test = False
scaling = True
k_fold = True

#process.start(raw_file_name, raw_file_path, 0.2, split_val_test and not k_fold, scaling)
logisticregression.start(raw_file_path, "TCGA-LUNG_", k_fold)

#process.start(raw_file_name, raw_file_path, 0.2, split_val_test and not k_fold, scaling)
#FCNN_training.start("Deep/", raw_file_path, "TCGA-LUNG_", split_val_test, k_fold)