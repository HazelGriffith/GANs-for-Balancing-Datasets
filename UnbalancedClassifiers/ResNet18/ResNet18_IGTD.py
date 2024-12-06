"""
@File: ResNet18 Training on Unbalanced IGTD Dataset
@Author: Puxin
@Last Modified: 2024-11-30
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
import torchvision.transforms as transforms 
from torchvision import datasets
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset_path = "igtd-16-train/IGTD_16_Train"
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
train_loader = DataLoader(train_dataset, batch_size=27, shuffle=False) 
test_loader = DataLoader(test_dataset, batch_size=27, shuffle=False)

import torchvision.models as models

model = models.resnet18(weights=None)

# Modify the final fully connected layer
model.fc = nn.Linear(model.fc.in_features, 1)

# Print the modified fc layer
print(model.fc)

#loss
loss_fn = nn.BCEWithLogitsLoss()

#optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.00021804075648719276, weight_decay=0.0001226417578986775) 

def test_phase(model, test_loader):
    model.eval()  
    test_loss = 0
    test_correct = 0
    
    with torch.no_grad():  
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long().squeeze(1)
            test_correct += (preds == labels).sum().item()
            
            loss = loss_fn(logits, labels.unsqueeze(1).float())
            test_loss += loss.item() * labels.size(0)
    
    # Compute average test loss and accuracy
    avg_test_loss = test_loss / len(test_loader.dataset)
    test_accuracy = test_correct / len(test_loader.dataset)
    return avg_test_loss, test_accuracy

num_epochs = 100
torch.manual_seed(2024)
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []
model.to(device)

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0
    train_correct = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels.unsqueeze(1).float())
        
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long().squeeze(1)
        train_correct += (preds == labels).sum().item()
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * labels.size(0)

    avg_train_loss = train_loss / len(train_loader.dataset)
    train_accuracy = train_correct / len(train_loader.dataset)

    # Test phase
    test_result = test_phase(model, test_loader)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.2f}, Test Loss: {test_result[0]:.2f}, Train Accuracy: {train_accuracy:.2f}, Test Accuracy: {test_result[1]:.2f}")
    train_losses.append(avg_train_loss)
    test_losses.append(test_result[0])
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_result[1])
    
    if test_result[0] < 0.2 or test_result[1] > 0.8:
        torch.save(model.state_dict(), f"ResNet18_igtd_ub_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), f"ResNet18_igtd_ub_epoch{epoch+1}.pt")