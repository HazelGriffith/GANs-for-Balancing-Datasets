# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 13:45:12 2024

@author: hazel
"""
import os
os.environ["TENSORBOARD_LOGGING"] = "1"
import torch
import torchvision
from torchvision.io import read_image
from torch.optim import Adam
import torch.utils.data as data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchgan
from torchgan.models import DCGANGenerator, DCGANDiscriminator
from torchgan.losses import MinimaxGeneratorLoss, MinimaxDiscriminatorLoss, WassersteinGradientPenalty, WassersteinGeneratorLoss
from torchgan.trainer import Trainer
from torch.utils.data import Dataset, DataLoader
import numpy as np

class DIDataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return 1465
        
    def __getitem__(self, idx):
        image_path = "../../../Data/PictData/DeepInsight/malign/sample"+str(idx+1)+".png"
        image = read_image(image_path).type(torch.FloatTensor)
        if self.transform:
            image = self.transform(image)
        return image, 1
        
    
def fetchDataLoader():
    train_dataloader = DataLoader(DIDataset(), batch_size=64, shuffle=True)
    return train_dataloader

trainer = Trainer({"generator": {"name": DCGANGenerator, "args": {"out_channels": 1, "step_channels": 2, 'label_type': 'required', 'out_size':64}, "optimizer": {"name": Adam, "args": {"lr": 0.0002, "betas": (0.5, 0.999)}}},
                   "discriminator": {"name": DCGANDiscriminator, "args": {"in_channels": 1, "step_channels": 2, 'label_type': 'required', 'in_size':64}, "optimizer": {"name": Adam, "args": {"lr": 0.0002, "betas": (0.5, 0.999)}}}},
                  [WassersteinGeneratorLoss(), WassersteinGradientPenalty()],
                  sample_size=64, epochs=400, device=torch.device('cuda:0'))

trainer(fetchDataLoader())

