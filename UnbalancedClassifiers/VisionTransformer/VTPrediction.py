# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 12:27:34 2024

@author: hazel
"""

import numpy as np
import pandas as pd
import os
os.environ["KERAS_BACKEND"] = "torch"
import torch
from torchvision.io import decode_image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import keras

class DIValDataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return 950
        
    def __getitem__(self, idx):
        image_path = "../../Data/PictData/DeepInsight/Validation/sample"+str(idx+1)+".png"
        image = decode_image(image_path).type(torch.FloatTensor)
        if self.transform:
            image = self.transform(image)
        return image

loaded_model = keras.saving.load_model("Saved_Models/UnbViT_model.keras")

validation_dataset = DIValDataset()
validation_data = []
for i in range(validation_dataset.__len__()):
    validation_data.append(validation_dataset.__getitem__(i))

validation_data = torch.stack(validation_data, dim=0)

prediction = loaded_model.__call__(validation_data).cpu().detach().numpy()

validation_csv = pd.read_csv("../../Data/TabData/Validation.csv")
validation_ids = np.expand_dims(validation_csv['ID'].to_numpy(dtype=int), axis=1)
predictionForKaggle = np.concatenate((validation_ids,prediction), axis=1)
predictionForKaggle_df = pd.DataFrame(predictionForKaggle, columns=['ID','Label'])
predictionForKaggle_df.to_csv("UnbVTPrediction.csv", index=False)