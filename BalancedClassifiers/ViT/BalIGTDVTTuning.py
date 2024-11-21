# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 13:55:17 2024

@author: hazel
"""

import numpy as np
import pandas as pd
import os
os.environ["KERAS_BACKEND"] = "torch"
#os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import torch
from torchvision.io import decode_image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
import keras
import keras_tuner as kt
from sklearn.model_selection import train_test_split
import winsound
print(torch.cuda.is_available())

class IGTDDataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.data = pd.read_csv("../../Data/PictData/IGTD/supervised.csv")
        
    def __len__(self):
        return 6000
    
    def __getTargets__(self):
        return self.data.loc[:,'class'].to_numpy()
        
    def __getitem__(self, idx):
        if (idx < 4465):
            filename = str(self.data.iloc[idx]['images'])
            label = self.data.iloc[idx]['class']
            image_path = "../../Data/PictData/IGTD/"+filename
        else:
            if (idx-4465)<10:
                filename = "00000" + str(idx-4465) + ".png"
            elif (idx-4465)<100:
                filename = "0000" + str(idx-4465) + ".png"
            elif (idx-4465)<1000:
                filename = "000" + str(idx-4465) + ".png"
            else:
                filename = "00" + str(idx-4465) + ".png"
            label = 1
            image_path = "../../Data/PictData/IGTD/CTGANMalignSynthData/"+filename
        image = decode_image(image_path).type(torch.FloatTensor)
        if self.transform:
            image = self.transform(image)
        return image, label
        
    
def fetchDataLoaders():
    data = IGTDDataset()
    targets = np.concatenate((data.__getTargets__(), np.ones(1535, dtype=np.int8)), axis=0, dtype=np.int8)
    trainInd, valInd,trainTargets,valTargets = train_test_split(
        range(data.__len__()),
        targets,
        stratify=targets,
        test_size=validation_size,
    )
    trainIndKT, valIndKT,_,_ = train_test_split(
        range(len(valInd)),
        valTargets,
        stratify=valTargets,
        test_size=validation_tuning_size,
    )
    train_split = Subset(data, trainInd)
    val_split = Subset(data, valInd)
    train_tuning_split = Subset(data, trainIndKT)
    val_tuning_split = Subset(data, valIndKT)
    
    train_dataloader = DataLoader(train_split, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_split, batch_size=batch_size, shuffle=False)
    train_tuning_dataloader = DataLoader(train_tuning_split, batch_size=batch_size, shuffle=True)
    val_tuning_dataloader = DataLoader(val_tuning_split, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, val_dataloader, train_tuning_dataloader, val_tuning_dataloader

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

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = keras.layers.Dense(units, activation=keras.activations.gelu)(x)
        x = keras.layers.Dropout(dropout_rate)(x)
    return x

def create_vit_classifier(hp):
    inputs = keras.Input(shape = (4, 16, 16))
    patches = Patches(patch_height, patch_width)(inputs)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    
    hp_transformerLayers = hp.Int('transformer_layers', min_value=6, max_value=22, step=2)
    hp_numHeads = hp.Int('number_of_heads', min_value=4, max_value=16, step=2)
    for i in range(hp_transformerLayers):
        x1 = keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        
        
        attention_output = keras.layers.MultiHeadAttention(
            num_heads=hp_numHeads, key_dim=projection_dim, dropout=0.1)(x1, x1)
        x2 = keras.layers.Add()([attention_output, encoded_patches])
        x3 = keras.layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        encoded_patches = keras.layers.Add()([x3, x2])
        
    representation = keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = keras.layers.Flatten()(representation)
    representation = keras.layers.Dropout(0.5)(representation)
    
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    logits = keras.layers.Dense(1)(features)
    
    model = keras.Model(inputs=inputs, outputs=logits)
    
    model.summary()
    
    hp_learningRate = hp.Float('learning_rate', min_value=1e-10, max_value=1e-1, step=10, sampling='log')
    hp_weightDecay = hp.Float('weight_decay', min_value=1e-10, max_value=1e-1, step=10, sampling='log')
    model.compile(
        loss='mean_squared_error',
        optimizer=keras.optimizers.Adam(learning_rate=hp_learningRate, weight_decay=hp_weightDecay),
        metrics=['root_mean_squared_error'],
    )
    return model

batch_size = 32
num_epochs = 20
image_height = 16
image_width = 16
patch_height = 2
patch_width = 2
num_patches = (image_height//patch_height)*(image_width//patch_width)
projection_dim = 64
transformer_units = [
    projection_dim * 2,
    projection_dim,
]
mlp_head_units = [
    2048,
    1024,
]

validation_size = 0.4
validation_tuning_size = 0.4
        
training_dataloader, validation_dataloader, training_tuning_dataloader, validation_tuning_dataloader = fetchDataLoaders()

#Tuning the model with Hyperband
tuner = kt.Hyperband(create_vit_classifier, objective='val_root_mean_squared_error', directory="/TuningResults/", overwrite=True)
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
]
tuner.search(training_tuning_dataloader, epochs=num_epochs, validation_data=validation_tuning_dataloader, callbacks=callbacks)
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
        
model = create_vit_classifier(best_hps)

history = model.fit(
    training_dataloader,
    batch_size=batch_size,
    epochs=300,
    validation_data=validation_dataloader,
    callbacks=callbacks,
)
        
model.save("Saved_Models/BalIGTDViT_model.keras")
        
hist_df = pd.DataFrame(history.history)
hist_df.to_csv("Saved_Models/BalIGTDViT_model.csv", sep=',')
print(best_hps.get('transformer_layers'))
print(best_hps.get('number_of_heads'))
print(best_hps.get('weight_decay'))
print(best_hps.get('learning_rate'))
keras.backend.clear_session()

duration = 1000
frequency = 440
winsound.Beep(frequency, duration)
