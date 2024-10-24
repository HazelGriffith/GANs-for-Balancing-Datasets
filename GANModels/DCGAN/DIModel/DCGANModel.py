# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 13:45:12 2024

@author: hazel
"""
import os
import torch
import torchvision
from torchvision.io import read_image
import torch.cpu as cpu
from torch.optim import Adam
import torch.utils.data as data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchgan
from torchgan.models import DCGANGenerator, DCGANDiscriminator
from torchgan.losses import MinimaxGeneratorLoss, MinimaxDiscriminatorLoss
from torchgan.trainer import Trainer
from torch.utils.data import Dataset, DataLoader
import numpy as np

class DIDataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.img_labels = np.loadtxt("../../../Data/PictData/trainingLabels.csv", delimiter=',').astype(np.float32)
        self.transform = transforms.Grayscale(num_output_channels=1)
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.img_labels)
        
    def __getitem__(self, idx):
        image_path = "../../../Data/PictData/DeepInsight/sample"+str(idx+1)+".png"
        image = read_image(image_path).type(torch.FloatTensor)
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
        
    
def fetchDataLoader():
    train_dataloader = DataLoader(DIDataset(), batch_size=64, shuffle=True)
    return train_dataloader

trainer = Trainer({"generator": {"name": DCGANGenerator, "args": {"out_channels": 1, "step_channels": 64, 'label_type': 'required', 'out_size':64}, "optimizer": {"name": Adam, "args": {"lr": 0.0002, "betas": (0.5, 0.999)}}},
                   "discriminator": {"name": DCGANDiscriminator, "args": {"in_channels": 1, "step_channels": 64, 'label_type': 'required', 'in_size':64}, "optimizer": {"name": Adam, "args": {"lr": 0.0002, "betas": (0.5, 0.999)}}}},
                  [MinimaxGeneratorLoss(), MinimaxDiscriminatorLoss()],
                  sample_size=64, epochs=20, device=cpu.current_device())

trainer(fetchDataLoader())
