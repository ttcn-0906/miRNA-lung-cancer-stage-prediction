import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from typing import List, Tuple
import matplotlib.pyplot as plt

class TrainDataset(Dataset):
    def __init__(self, features, labels):
        self.features, self.labels = features, labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature, label
    
def load_train_dataset(file_path: str, feature_list: list)->Tuple[List, List]:
    predict_labels = ["Stage"]

    df = pd.read_csv(file_path)
    labels = df[predict_labels].values.tolist()

    all_columns = df.columns.tolist()
    if not feature_list == None:
        mirna_cols = feature_list
    else:
        mirna_cols = [col for col in all_columns if col.startswith("hsa-")]

    df = df[mirna_cols]
    features = df.values.tolist()
    #features has length 1535 mirna
    
    return features, labels

def load_feature_list(feature_file: str):
    df = pd.read_csv(feature_file)

    feature_list = df["selected features"].values.tolist()
    feature_list = feature_list[0:200]

    return feature_list

def plot(file_path: str, train_losses: List, val_losses: List):
    # (TODO) Plot the training loss and validation loss of CNN, and save the plot to 'loss.png'
    #        xlabel: 'Epoch', ylabel: 'Loss'

    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(file_path + 'Loss/loss.png')
    plt.close()

    #raise NotImplementedError

    print("Save the plot to 'loss.png'")
    return