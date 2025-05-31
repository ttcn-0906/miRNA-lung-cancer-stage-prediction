import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, ConfusionMatrixDisplay, f1_score, balanced_accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

K_FOLD = 5
RANDOM_STATE = 42
TOP_FEATURES = 200
# === 1. Data Preprocessing ===
def load_data(file_path: str, file_prefix: str) -> tuple[pd.DataFrame, list, pd.DataFrame, list]:

    required_labels = [
        "Age", "Sex"
    ]
    predict_labels = ["Stage"]

    train = pd.read_csv(file_path + file_prefix + "train.csv")
    test = pd.read_csv(file_path + file_prefix + "validate.csv")

    y_train = [labels[0] for labels in train[predict_labels].values.tolist()]
    y_test = [labels[0] for labels in test[predict_labels].values.tolist()]

    all_columns = train.columns.tolist()
    mirna_cols = [col for col in all_columns if col.startswith("hsa-")]

    train = train[mirna_cols + required_labels]
    test = test[mirna_cols + required_labels]
    
    X_train = train
    X_test = test

    return X_train, y_train, X_test, y_test

# === 2. Feature Selection ===
def select_features(X_train: pd.DataFrame, y_train: list, X_test: pd.DataFrame):
    """
    consider only mirna feature but metadata
    """
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

# === 3. Model Training (Random Forest) ===
def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    cv = StratifiedKFold(n_splits=K_FOLD, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
    print(f"Cross-validated F1 scores: {scores}")
    print(f"Mean F1: {scores.mean():.3f}")

    model.fit(X_train, y_train)
    return model

# === 4. Evaluation ===
def evaluate_model(model, X_test, y_test, result_path):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("Performance:")
    print(f" - AUC-ROC Score       : {roc_auc_score(y_test, y_proba):.3f}")
    print(f" - F1 Score            : {f1_score(y_test, y_pred):.3f}")
    print(f" - Balanced Accuracy   : {balanced_accuracy_score(y_test, y_pred):.3f}")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Early", "Late"]))

    os.makedirs(result_path + "results/", exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test,
        cmap="Blues",
        display_labels=["Early", "Late"],
        ax=ax
    )
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{result_path}results/confusion_matrix_rf.png", dpi=300)
    plt.close(fig)

# === 5. Save Results ===
def save_results(model, features, result_path):
    #scaled using DataProcess/process
    os.makedirs(result_path + "results/", exist_ok=True)
    joblib.dump(model, f"{result_path}results/rf_model.pkl")
    pd.Series(features, name="selected features").to_csv(f"{result_path}results/rf_selected_features.csv", index=False)

# === 6. Main Pipeline ===
def start(file_path: str = "../DataProcess/", file_prefix: str = "TCGA-LUNG_", result_path: str = "./"):
    """
    RF model does not support load_all option right now. it uses validate_data as testing data for evaluation.
    """
    print("Starting Random Forest pipeline...")
    X_train, y_train, X_test, y_test = load_data(file_path, file_prefix)

    meta_cols = ["Age", "Sex"]

    print("Selecting features...")
    X_train_sel, X_test_sel, selected = select_features(X_train, y_train, X_test)

    #add metadata back
    X_train_full = np.concatenate([X_train_sel, X_train[meta_cols].to_numpy()], axis=1)
    X_test_full = np.concatenate([X_test_sel, X_test[meta_cols].to_numpy()], axis=1)

    print("Training Random Forest model...")
    model = train_random_forest(X_train_full, y_train)

    print("Evaluating model...")
    all_features = list(selected) + meta_cols
    evaluate_model(model, X_test_full, y_test, result_path)

    print("Saving model and results...")
    save_results(model, all_features, result_path)

    print("Random Forest pipeline completed.")

if __name__ == "__main__":
    start()