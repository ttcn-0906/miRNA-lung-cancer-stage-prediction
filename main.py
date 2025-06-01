from DataProcess import process
from Deep import FCNN_training
from Baseline import logisticregression
from Ml import RandomForest, XGBoost

#data process
raw_file_path = "DataProcess/"
raw_file_name = "TCGA-LUNG_raw_data.csv"
feature_file = "Ml/results/selected_features.csv"
split_val_test = True
scaling = True
k_fold = True
select_k_best = True #use features selected in XGBoost.py for FCNN

#process.start(raw_file_name, raw_file_path, 0.2, split_val_test and not k_fold, scaling)
#logisticregression.start(raw_file_path, "TCGA-LUNG_", k_fold)

#process.start(raw_file_name, raw_file_path, 0.1, split_val_test and not k_fold, scaling)
#RandomForest.start(raw_file_path, "TCGA-LUNG_", "Ml/")
#XGBoost.start(raw_file_path, "TCGA-LUNG_", "Ml/", False)

process.start(raw_file_name, raw_file_path, 0.1, split_val_test and not k_fold, scaling)
FCNN_training.start("Deep/", raw_file_path, "TCGA-LUNG_", split_val_test, k_fold, select_k_best, feature_file)