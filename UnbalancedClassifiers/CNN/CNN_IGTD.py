"""
@File: CNN Training on Unbalanced IGTD Dataset
@Author: Puxin
@Last version: 2024-12-02
"""

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  

set_seed(2024)

import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  
    transforms.Resize((64, 64)),                  
    transforms.ToTensor(),                        
])

dataset_path = "/igtd-16-train/IGTD_16_Train"
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

from sklearn.model_selection import train_test_split
labels = dataset.targets

train_idx, test_idx = train_test_split(
    np.arange(len(labels)),
    test_size=0.2,
    random_state=2024,
    shuffle=True,
    stratify=labels)

print(f"Training set count: {len(train_idx)}")
print(f"test set count: {len(test_idx)}")

from torch.utils.data import Subset
# Subset dataset for train and test
train_dataset = Subset(dataset, train_idx)
test_dataset= Subset(dataset, test_idx)

# Dataloader for train and test
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False) 
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

import torch.nn.functional as F

model = nn.Sequential(
    # Features
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=0),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            
    # Classifier
    nn.Flatten(),
    nn.Linear(64 * 15 * 15, 128),
    nn.ReLU(),
    nn.Linear(128, 1)).to(device)

#loss
loss_fn = nn.BCEWithLogitsLoss() 

#optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 

def test_loss(model, test_loader):
    model.eval()  
    test_loss = 0.0
    
    with torch.no_grad():  
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            logits = model(images)
            loss = loss_fn(logits, labels.unsqueeze(1).float())
            test_loss += loss.item() * labels.size(0)
    
    avg_test_loss = test_loss / len(test_loader.dataset)
    return avg_test_loss

num_epochs = 100
torch.manual_seed(2024)
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels.unsqueeze(1).float())
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * labels.size(0)

    avg_train_loss = train_loss / len(train_loader.dataset)
    
    # Test phase
    avg_test_loss = test_loss(model, test_loader)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")
    train_losses.append(avg_train_loss)
    test_losses.append(avg_test_loss)
    
    if avg_test_loss < 0.25:
        torch.save(model.state_dict(), f'cnn_IGTD_UB_epoch{epoch+1}.pth')
        torch.save(model.state_dict(), f'cnn_IGTD_UB_epoch{epoch+1}.pt') 