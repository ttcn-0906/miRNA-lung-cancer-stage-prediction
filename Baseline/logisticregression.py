import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, make_scorer

PENALTY = 'l2'
SOLVER = 'liblinear'
C = 1e-3
K_FOLD = 10
RANDOM_STATE = 42

def load_data(file_path: str, file_prefix: str, load_all: bool = False):

    required_labels = [
        "Age", "Sex"
    ]
    predict_labels = ["Stage"]

    if load_all:
        train = pd.read_csv(file_path + file_prefix + "all.csv")
        y_train = [labels[0] for labels in train[predict_labels].values.tolist()]

        all_columns = train.columns.tolist()
        mirna_cols = [col for col in all_columns if col.startswith("hsa-")]

        train = train[mirna_cols + required_labels]
        X_train = train.values.tolist()

        X_train, y_train = shuffle(X_train, y_train, random_state=RANDOM_STATE)

        return X_train, y_train, None, None

    train = pd.read_csv(file_path + file_prefix + "train.csv")
    test = pd.read_csv(file_path + file_prefix + "validate.csv")

    y_train = [labels[0] for labels in train[predict_labels].values.tolist()]
    y_test = [labels[0] for labels in test[predict_labels].values.tolist()]

    all_columns = train.columns.tolist()
    mirna_cols = [col for col in all_columns if col.startswith("hsa-")]

    train = train[mirna_cols + required_labels]
    test = test[mirna_cols + required_labels]
    
    X_train = train.values.tolist()
    X_test = test.values.tolist()

    return X_train, y_train, X_test, y_test

def train_fix_split(file_path: str, file_prefix: str):

    X_train, y_train, X_test, y_test = load_data(file_path, file_prefix)

    clf = LogisticRegression(penalty=PENALTY, solver=SOLVER, C=C, max_iter=10000, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=4))
    print("===== Confusion Matrix ======")
    print(confusion_matrix(y_test, y_pred))
    print("========= Accuracy ==========")
    print("Accuracy:", accuracy_score(y_test, y_pred))

def train_k_fold(file_path: str, file_prefix: str):

    X_train, y_train, _, _ = load_data(file_path, file_prefix, load_all=True)

    total_acc = 0
    max_acc = 0
    cr = None
    cm = None

    skf = StratifiedKFold(n_splits=K_FOLD, shuffle=True, random_state=RANDOM_STATE)
    for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
        
        fold_X_train = [X_train[i] for i in train_index]
        fold_y_train = [y_train[i] for i in train_index]

        fold_X_test = [X_train[i] for i in val_index]
        fold_y_test = [y_train[i] for i in val_index]

        clf = LogisticRegression(penalty=PENALTY, solver=SOLVER, C=C, max_iter=10000, random_state=RANDOM_STATE)
        clf.fit(fold_X_train, fold_y_train)

        fold_y_pred = clf.predict(fold_X_test)
        acc = accuracy_score(fold_y_test, fold_y_pred)
        total_acc += acc

        if acc > max_acc:
            max_acc = acc
            cr = classification_report(fold_y_test, fold_y_pred, digits=4)
            cm = confusion_matrix(fold_y_test, fold_y_pred)

    print("=== Classification Report for best LR ===")
    print(cr)
    print("===== Confusion Matrix for best LR ======")
    print(cm)
    print("========= Accuracy for best LR ==========")
    print("Accuracy:", max_acc)
    print("=========== Average Accuracy ============")
    print("Accuracy:", total_acc/K_FOLD)

def grid_search_lr_finding(file_path: str, file_prefix: str):

    X_all, y_all, _, _ = load_data(file_path, file_prefix, load_all=True)

    skf = StratifiedKFold(n_splits=K_FOLD, shuffle=True, random_state=RANDOM_STATE)

    model = LogisticRegression(random_state=RANDOM_STATE, max_iter=10000)

    param_grid = {
        'penalty': ['l1', 'l2'],
        'C': [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2],
        'solver': ['liblinear'],
    }
    scorer = make_scorer(f1_score)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=skf,
        scoring=scorer,
        n_jobs=-1 #use all processor
    )

    grid_search.fit(X_all, y_all)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation F1-score: {grid_search.best_score_}")

    best_lr_model = grid_search.best_estimator_

    y_pred_best = cross_val_predict(best_lr_model, X_all, y_all, cv=skf, n_jobs=-1)
    print("=== Classification Report for best LR ===")
    print(classification_report(y_all, y_pred_best))

def start(file_path: str = "../DataProcess/", file_prefix: str = "TCGA-LUNG_", k_fold: bool = False, grid_search: bool = False):
    """
    if k fold is enabled, LR model use all_data as training data. validate_data and test_data is ignored.
    """
    if grid_search:
        grid_search_lr_finding(file_path, file_prefix)
    elif k_fold:
        train_k_fold(file_path, file_prefix)
    else:
        train_fix_split(file_path, file_prefix)
    
if __name__ == '__main__':
    start(grid_search=False)