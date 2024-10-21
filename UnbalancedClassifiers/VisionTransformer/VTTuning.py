# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 11:54:48 2024

@author: hazel
"""

import numpy as np
import pandas as pd
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import keras
import keras_tuner as kt

class Patches(keras.layers.Layer):
    def __init__(self, patch_height, patch_width):
        super().__init__()
        self.patch_height = patch_height
        self.patch_width = patch_width
        
    def call(self, images):
        input_shape = keras.ops.shape(images)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        num_patches_h = height // self.patch_height
        num_patches_w = width // self.patch_width
        patches = keras.ops.image.extract_patches(images, size=(self.patch_height, self.patch_width))
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

class PatchEncoder(keras.layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
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
        config.update({"num_patches": self.num_patches})
        return config

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = keras.layers.Dense(units, activation=keras.activations.gelu)(x)
        x = keras.layers.Dropout(dropout_rate)(x)
    return x

def create_vit_classifier(hp):
    inputs = keras.Input(shape = (12, 12, 1))
    patches = Patches(patch_height, patch_width)(inputs)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    
    hp_transformerLayers = hp.Int('transformer_layers', min_value=6, max_value=10, step=2)
    hp_numHeads = hp.Int('number_of_heads', min_value=4, max_value=8, step=2)
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
    
    hp_learningRate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, step=10, sampling='log')
    hp_weightDecay = hp.Float('weight_decay', min_value=1e-5, max_value=1e-3, step=10, sampling='log')
    model.compile(
        loss='mean_squared_error',
        optimizer=keras.optimizers.Adam(learning_rate=hp_learningRate, weight_decay=hp_weightDecay),
        metrics=['root_mean_squared_error'],
    )
    return model

batch_size = 56
num_epochs = 20
image_height = 12
image_width = 12
patch_height = 3
patch_width = 3
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

validation_size = int(0.2*4465)
        
training_data = np.loadtxt("../../Data/PictData/DITrainingData.csv", delimiter=',').astype(int)
training_labels = np.loadtxt("../../Data/PictData/trainingLabels.csv", delimiter=',').astype(int)



training_data = np.expand_dims(training_data.reshape(4465,12,12),-1)
validation_data = training_data[:validation_size]
validation_labels = training_labels[:validation_size]


#Tuning the model with Hyperband
tuner = kt.Hyperband(create_vit_classifier, objective='val_root_mean_squared_error', directory="/TuningResults/", overwrite=True)
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
    ]
tuner.search(validation_data, validation_labels, epochs=num_epochs, validation_split=0.2, callbacks=callbacks)
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
        
model = create_vit_classifier(best_hps)
        
history = model.fit(
    training_data,
    training_labels,
    batch_size=batch_size,
    epochs=100,
    validation_data=(validation_data, validation_labels),
    callbacks=callbacks,
    )
        
model.save("Saved_Models/UnbViT_model.keras")
        
hist_df = pd.DataFrame(history.history)
hist_df.to_csv("Saved_Models/UnbViT_model.csv", sep=',')
        
keras.backend.clear_session()
