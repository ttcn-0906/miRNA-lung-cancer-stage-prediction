import os
import joblib
import torch
import pandas as pd
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, ConfusionMatrixDisplay, f1_score, balanced_accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from xgboost import XGBClassifier

K_FOLD = 5
RANDOM_STATE = 42
TOP_FEATURES = 200

# === 1. Data Preprocessing ===
def load_data(file_path: str, file_prefix: str) -> tuple[pd.DataFrame, list, pd.DataFrame, list]:

    predict_labels = ["Stage"]

    train = pd.read_csv(file_path + file_prefix + "train.csv")
    test = pd.read_csv(file_path + file_prefix + "validate.csv")

    y_train = [labels[0] for labels in train[predict_labels].values.tolist()]
    y_test = [labels[0] for labels in test[predict_labels].values.tolist()]

    all_columns = train.columns.tolist()
    mirna_cols = [col for col in all_columns if col.startswith("hsa-")]

    train = train[mirna_cols]
    test = test[mirna_cols]
    
    X_train = train
    X_test = test

    return X_train, y_train, X_test, y_test

# === 2. Feature Selection ===
def select_features(X_train: pd.DataFrame, y_train: list, X_test: pd.DataFrame):
    mirna_cols = [col for col in X_train.columns if col.startswith("hsa-")]
    X_train_miRNA = X_train[mirna_cols]
    X_test_miRNA = X_test[mirna_cols]

    constant_cols = X_train_miRNA.columns[X_train_miRNA.nunique() <= 1]
    X_train_miRNA = X_train_miRNA.drop(columns=constant_cols)
    X_test_miRNA = X_test_miRNA.drop(columns=constant_cols)

    selector = SelectKBest(f_classif, k=min(TOP_FEATURES, X_train_miRNA.shape[1]))
    X_train_sel = selector.fit_transform(X_train_miRNA, y_train)
    X_test_sel = selector.transform(X_test_miRNA)
    selected = X_train_miRNA.columns[selector.get_support()]

    return X_train_sel, X_test_sel, selected

# === 3. Model Training ===
def find_model_param(X_train, y_train, X_val, y_val):
    scale_pos_weight = sum(i==0 for i in y_train) / sum(i==1 for i in y_train)

    param_grid = {
        'max_depth': [3, 4],
        'learning_rate': [0.01],
        'n_estimators': [200, 500, 1000],
        'gamma': [0, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'scale_pos_weight': [scale_pos_weight]
    }

    model = XGBClassifier(
        device='cpu',
        objective='binary:logistic',
        random_state=RANDOM_STATE,
        eval_metric='logloss',
        early_stopping_rounds=100
    )

    grid = GridSearchCV(
        model,
        param_grid=param_grid,
        scoring='f1',
        cv=StratifiedKFold(K_FOLD, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
    print("Best hyperparameters:")
    for k, v in grid.best_params_.items():
        print(f" - {k}: {v}")
    return grid.best_estimator_

def train_model(X_all, y_all):
    kf = StratifiedKFold(n_splits=K_FOLD, shuffle=True, random_state=RANDOM_STATE)

    xgb_results = {
        'accuracy': [],
        'f1_score': [],
        'roc_auc': [],
        'classification_report': None
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_f1 = 0
    best_model = None

    for fold, (train_index, val_index) in enumerate(kf.split(X_all, y_all)):
        X_train, X_val = np.array([X_all[i] for i in train_index]), np.array([X_all[i] for i in val_index])
        y_train, y_val = np.array([y_all[i] for i in train_index]), np.array([y_all[i] for i in val_index])

        np_X_train, np_X_val = X_train, X_val
        np_y_train, np_y_val = y_train, y_val

        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train, dtype=torch.long).to(device)
        X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val = torch.tensor(y_val, dtype=torch.long).to(device)

        print(f"=== Fold {fold+1}/{K_FOLD} ===")

        model = XGBClassifier(
            device="cuda" if torch.cuda.is_available() else "cpu",
            objective='binary:logistic',
            eval_metric='logloss',
            n_estimators=500,
            learning_rate=0.01,
            gamma=0,
            max_depth=3,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_weight=1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            early_stopping_rounds=100
        )

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        acc = accuracy_score(np_y_val, y_pred)
        f1 = f1_score(np_y_val, y_pred)
        roc_auc = roc_auc_score(np_y_val, y_prob)
        report = classification_report(np_y_val, y_pred)

        if f1 > max_f1:
            max_f1 = f1
            best_model = model
            xgb_results['classification_report'] = report

        print(f"XGBoost - Acc: {acc:.4f}, F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}")

        xgb_results['accuracy'].append(acc)
        xgb_results['f1_score'].append(f1)
        xgb_results['roc_auc'].append(roc_auc)

    print("=== XGBoost K-Fold Average Result ===")
    print(f"Average Accuracy: {np.mean(xgb_results['accuracy']):.4f}")
    print(f"Average F1-Score: {np.mean(xgb_results['f1_score']):.4f}")
    print(f"Average ROC AUC: {np.mean(xgb_results['roc_auc']):.4f}")
    print(f"Best classification report: \n{xgb_results['classification_report']}")

    return best_model

# === 4. Evaluation ===
def evaluate_model(model: XGBClassifier, X_test, y_test, result_path):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("Performance:")
    print(f" - AUC-ROC Score       : {roc_auc_score(y_test, y_proba):.3f}")
    print(f" - F1 Score            : {f1_score(y_test, y_pred):.3f}")
    print(f" - Balanced Accuracy   : {np.array(balanced_accuracy_score(y_test, y_pred)):.3f}")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Early", "Late"]))

    os.makedirs(result_path + "results/", exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="Blues", display_labels=["Early", "Late"], ax=ax)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{result_path}results/confusion_matrix.png", dpi=300)
    plt.close(fig)

# === 5. Save ===
def save_results(model: XGBClassifier, features, result_path):
    #scaled using DataProcess/process
    os.makedirs(result_path + "results/", exist_ok=True)
    joblib.dump(model, f"{result_path}results/best_model.pkl")
    pd.Series(features, name="selected features").to_csv(f"{result_path}results/selected_features.csv", index=False)

    booster = model.get_booster()
    importance = booster.get_score(importance_type='gain')
    importance_list = []
    for i in range(len(features)):
        if f"f{i}" in importance.keys():
            importance_list.append(importance[f"f{i}"])
        else:
            importance_list.append(0.0)
    importance_df = pd.DataFrame({
        'selected features': features,
        'importance': importance_list
    }).sort_values(by='importance', ascending=False)

    print(importance_df.head(10))

    importance_df.to_csv(result_path + "results/XGBimportance.csv", index=False)

# === 6. Main ===
def start(file_path: str = "../DataProcess/", file_prefix: str = "TCGA-LUNG_", result_path: str = "./", grid_search: bool = False):
    """
    XGB model does not support load_all option right now. it uses validate_data as testing data for evaluation.
    train/val split size should be smaller for larger training data.
    """
    print("Starting pipeline...")
    X_train, y_train, X_test, y_test = load_data(file_path, file_prefix)

    print("Selecting features...")
    X_train_sel, X_test_sel, selected = select_features(X_train, y_train, X_test)

    print("Training model...")
    if grid_search:
        model = find_model_param(X_train_sel, y_train, X_test_sel, y_test)
    else:
        model = train_model(X_train_sel, y_train)

    print("Evaluating model...")
    all_features = list(selected)
    if not grid_search: #uses cuda
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_test_sel = torch.tensor(X_test_sel, dtype=torch.float32).to(device)

    evaluate_model(model, X_test_sel, y_test, result_path)

    print("Saving...")
    save_results(model, all_features, result_path)

    print("Pipeline complete.")

if __name__ == "__main__":
    start(grid_search=True)