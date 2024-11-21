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

@keras.saving.register_keras_serializable()
class Patches(keras.layers.Layer):
    def __init__(self, patch_height, patch_width, **kwargs):
        super().__init__(**kwargs)
        self.patch_height = patch_height
        self.patch_width = patch_width
        
    def call(self, images):
        input_shape = keras.ops.shape(images)
        batch_size = input_shape[0]
        channels = input_shape[1]
        height = input_shape[2]
        width = input_shape[3]
        
        num_patches_h = height // self.patch_height
        num_patches_w = width // self.patch_width
        patches = keras.ops.image.extract_patches(images, size=(self.patch_height, self.patch_width), data_format="channels_first")
        patches = keras.ops.reshape(
            patches,
            (
                batch_size,
                num_patches_h * num_patches_w,
                self.patch_height * self.patch_width * channels,
            ),
        )
        return patches
    
    def get_config(self):
        config = super().get_config()
        config.update({"patch_height": self.patch_height, 
                       "patch_width": self.patch_width})
        return config

@keras.saving.register_keras_serializable()
class PatchEncoder(keras.layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = keras.layers.Dense(units=projection_dim)
        self.position_embedding = keras.layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )
        
    def call(self, patch):
        positions = keras.ops.expand_dims(
            keras.ops.arange(start=0, stop=self.num_patches, step=1), axis=0)
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded
    
    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches,
                       "projection_dim": self.projection_dim})
        return config


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

loaded_model = keras.saving.load_model("Saved_Models/BalDIViT_model.keras")

validation_dataset = DIValDataset()
validation_data1 = []
validation_data2 = []
for i in range(475):
    validation_data1.append(validation_dataset.__getitem__(i))
    validation_data2.append(validation_dataset.__getitem__(i+475))
validation_data1 = torch.stack(validation_data1, dim=0)
validation_data2 = torch.stack(validation_data2, dim=0)

prediction1 = loaded_model.__call__(validation_data1).cpu().detach().numpy()
prediction2 = loaded_model.__call__(validation_data2).cpu().detach().numpy()
prediction = np.concatenate((prediction1, prediction2), axis=0)

keras.backend.clear_session()
validation_csv = pd.read_csv("../../Data/TabData/Validation.csv")
validation_ids = np.expand_dims(validation_csv['ID'].to_numpy(dtype=int), axis=1)
predictionForKaggle = np.concatenate((validation_ids,prediction), axis=1)
predictionForKaggle_df = pd.DataFrame(predictionForKaggle, columns=['ID','Label'])
predictionForKaggle_df.to_csv("BalDIVTPrediction.csv", index=False)
