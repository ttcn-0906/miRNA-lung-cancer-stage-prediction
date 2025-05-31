import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from loguru import logger
try:
    from .FCNN import FCNN, train, validate, test
    from .utils import TrainDataset, load_train_dataset, plot
except ImportError:
    from FCNN import FCNN, train, validate, test
    from utils import TrainDataset, load_train_dataset, plot

BATCH_SIZE = 32
INPUT_DIM = 1537
OUTPUT_SIZE = 2
LR = 1e-5
WEIGHT_DECAY = 1e-3
EPOCHS = 100
K_FOLD = 5
RANDOM_STATE = 42

def load_data(file_path: str, file_prefix: str, split_val_test: bool = False, load_all: bool = False) -> \
    tuple[list, list, list | None, list | None, list | None, list | None]:
    """
    If ``load_all = True``, data_all is loaded as train data and ``split_val_test`` is ignored
    """
    logger.info("Start loading data")
    if load_all:
        train_features, train_labels = load_train_dataset(file_path + file_prefix + "all.csv")
        train_features, train_labels = shuffle(train_features, train_labels, random_state=RANDOM_STATE)
        
        return train_features, train_labels, None, None, None, None
    
    train_features, train_labels = load_train_dataset(file_path + file_prefix + "train.csv")
    val_features, val_labels = load_train_dataset(file_path + file_prefix + "validate.csv")
    if split_val_test:
        test_features, test_labels = load_train_dataset(file_path + file_prefix + "test.csv")
    else:
        test_features, test_labels = None, None

    return  train_features, train_labels, val_features, val_labels, test_features, test_labels

def train_fix_split(model_path: str ="./", file_path: str = "../DataProcess/", file_prefix: str = "TCGA-LUNG_", split_val_test: bool = False):
    #load data
    train_features, train_labels, val_features, val_labels, test_features, test_labels = load_data(file_path, file_prefix, split_val_test)

    train_dataset = TrainDataset(train_features, train_labels)
    val_dataset = TrainDataset(val_features, val_labels)
    if split_val_test:
        test_dataset = TrainDataset(test_features, test_labels)
    
    #training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Start training FCNN")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = FCNN(INPUT_DIM, OUTPUT_SIZE).to(device)
    #able to handle multi-stage classification
    criterion = nn.CrossEntropyLoss()

    # Optimizer configuration
    base_params = [param for name, param in model.named_parameters() if param.requires_grad]
    optimizer = optim.Adam(base_params, lr=LR, weight_decay=WEIGHT_DECAY)

    train_losses = []
    val_losses = []
    max_acc = 0

    for epoch in range(EPOCHS):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        #print epoch result and save model
        logger.info(f"Epoch [{epoch+1}/{EPOCHS}] - Training Loss: {train_loss:.4f} - Validation Loss: {val_loss:.4f} - Validation Accuracy: {val_acc:.4f}")
        
        if val_acc > max_acc:
            max_acc = val_acc
            torch.save(model.state_dict(), model_path + "Model/best_model.pth")
            logger.info(f"Saving model with validation accuracy: {val_acc:.4f}")

    logger.info(f"Best Accuracy: {max_acc:.4f}")

    #plot losses
    plot(model_path, train_losses, val_losses)

    #test
    if split_val_test:
        model.load_state_dict(torch.load('best_model.pth'))
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test(model, test_loader, criterion, device)

def train_k_fold(model_path: str ="./", file_path: str = "../DataProcess/", file_prefix: str = "TCGA-LUNG_", test_data: bool = False):
    #load data
    if test_data:
        #validate data used as test data
        train_features, train_labels, test_features, test_labels, _, _ = load_data(file_path, file_prefix, split_val_test=False)
    else:
        train_features, train_labels, _, _, _, _ = load_data(file_path, file_prefix, load_all=True)

    #train and validate dataset is created inside k fold loop
    if test_data:
        test_dataset = TrainDataset(test_features, test_labels)
    
    #training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Start training FCNN with K-Fold")

    #max_accuracy is now across folds
    max_acc = 0
    train_losses = [0 for _ in range(EPOCHS)]
    val_losses = [0 for _ in range(EPOCHS)]

    skf = StratifiedKFold(n_splits=K_FOLD, shuffle=True, random_state=RANDOM_STATE)
    for fold, (train_index, val_index) in enumerate(skf.split(train_features, train_labels)):
        logger.info(f"Fold: {fold+1}/{K_FOLD}")

        fold_train_features = [train_features[i] for i in train_index]
        fold_train_labels = [train_labels[i] for i in train_index]

        fold_val_features = [train_features[i] for i in val_index]
        fold_val_labels = [train_labels[i] for i in val_index]

        train_dataset = TrainDataset(fold_train_features, fold_train_labels)
        val_dataset = TrainDataset(fold_val_features, fold_val_labels)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = FCNN(INPUT_DIM, OUTPUT_SIZE).to(device)
        #able to handle multi-stage classification
        criterion = nn.CrossEntropyLoss()

        # Optimizer configuration
        base_params = [param for name, param in model.named_parameters() if param.requires_grad]
        optimizer = optim.Adam(base_params, lr=LR, weight_decay=WEIGHT_DECAY)

        for epoch in range(EPOCHS):
            train_loss = train(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            train_losses[epoch] += train_loss
            val_losses[epoch] += val_loss

            #print epoch result and save model
            logger.info(f"Epoch [{epoch+1}/{EPOCHS}] - Training Loss: {train_loss:.4f} - Validation Loss: {val_loss:.4f} - Validation Accuracy: {val_acc:.4f}")
            
            if val_acc > max_acc:
                max_acc = val_acc
                torch.save(model.state_dict(), model_path + "Model/best_model.pth")
                logger.info(f"Saving model with validation accuracy: {val_acc:.4f}")

    logger.info(f"Best Accuracy: {max_acc:.4f}")

    train_losses = [train_losses[i]/K_FOLD for i in range(EPOCHS)]
    val_losses = [val_losses[i]/K_FOLD for i in range(EPOCHS)]

    #plot losses
    plot(model_path, train_losses, val_losses)

    #test
    if test_data:
        model.load_state_dict(torch.load('best_model.pth'))
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test(model, test_loader, criterion, device)

def start(model_path: str ="./", file_path: str = "../DataProcess/", file_prefix: str = "TCGA-LUNG_", split_val_test: bool = False, k_fold: bool = True):
    """
    When Stratified K-Fold is selected and ``split_val_test = False``, the model use data_all in training without final test data.
    If ``split_val_test = True``, the model use data_train in training and use data_validate for final test.
    """
    if k_fold:
        train_k_fold(model_path, file_path, file_prefix, split_val_test)
    else:
        train_fix_split(model_path, file_path, file_prefix, split_val_test)

if __name__ == '__main__':
    start()