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
    
def load_train_dataset(file_path: str)->Tuple[List, List]:
    required_labels = [
        "ID", "Age", "Stage",
        "Overall Survival (Months)", "Overall Survival Status", "Sex",
        "Subtype"
    ]
    predict_labels = ["Stage"]

    df = pd.read_csv(file_path)
    labels = df[predict_labels].values.tolist()

    df = df.drop(columns=required_labels, errors="ignore")
    features = df.values.tolist()
    #features has length 1535 mirna
    
    return features, labels

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
    plt.savefig(file_path + 'loss.png')
    plt.close()

    #raise NotImplementedError

    print("Save the plot to 'loss.png'")
    return