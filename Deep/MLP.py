import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple
import pandas as pd

class MLP(nn.Module):
    def __init__(self, num_dim, num_classes):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(num_dim, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)

        self.stage_output = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.dropout1(self.relu1(self.bn1(self.fc1(x))))
        x = self.dropout2(self.relu2(self.bn2(self.fc2(x))))
        x = self.stage_output(x)
    
        return x

def train(model: MLP, train_loader: DataLoader, criterion, optimizer, device)->float:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    
    for inputs, label_batches in progress_bar:
        inputs, label_batches = inputs.to(device), label_batches.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        
        stage_labels = label_batches[:, 0]
        loss = criterion(outputs, stage_labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
        _, predicted = torch.max(outputs, 1)
        total += stage_labels.size(0)
        correct += (predicted == stage_labels).sum().item()
        
        progress_bar.set_postfix({
            "batch_loss": loss.item(),
            "batch_acc": (predicted == stage_labels).sum().item() / stage_labels.size(0)
        })
    
    avg_loss = running_loss / len(train_loader.dataset)

    return avg_loss


def validate(model: MLP, val_loader: DataLoader, criterion, device)->Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(val_loader, desc="Validating", leave=False)
    
    with torch.no_grad():
        for inputs, label_batches in progress_bar:
            inputs, label_batches = inputs.to(device), label_batches.to(device)
            
            outputs = model(inputs)
            stage_labels = label_batches[:, 0]
            loss = criterion(outputs, stage_labels)
            
            running_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs, 1)
            total += stage_labels.size(0)
            correct += (predicted == stage_labels).sum().item()
            
            progress_bar.set_postfix({
                "batch_loss": loss.item(),
                "batch_acc": (predicted == stage_labels).sum().item() / stage_labels.size(0)
            })
    
    avg_loss = running_loss / len(val_loader.dataset)
    accuracy = correct / total
    
    return avg_loss, accuracy

def test(model: MLP, test_loader: DataLoader, criterion, device):
    model.eval()
    results = []
    total = 0
    correct = 0

    with torch.no_grad():
        for inputs, label_batches in tqdm(test_loader, desc="Testing"):
            inputs, label_batches = inputs.to(device), label_batches.to(device)

            outputs = model(inputs)
            stage_labels = label_batches[:, 0]

            _, predicted = torch.max(outputs, 1)
            total += stage_labels.size(0)
            correct += (predicted == stage_labels).sum().item()

    accuracy = correct / total

    return accuracy