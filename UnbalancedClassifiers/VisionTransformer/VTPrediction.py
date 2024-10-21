# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 12:27:34 2024

@author: hazel
"""

import numpy as np
import pandas as pd
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import keras

validation_data = pd.read_csv("../../Data/TabData/Validation.csv")
validation_ids = np.expand_dims(validation_data['ID'].to_numpy(), axis=1)
validation_data = validation_data.drop(labels='ID',axis=1).to_numpy()

loaded_model = keras.saving.load_model("Saved_Models/UnbViT_model.keras")

prediction = np.expand_dims(loaded_model.predict(validation_data), axis=1)

predictionForKaggle = np.concatenate((validation_ids,prediction), axis=1)
predictionForKaggle_df = pd.DataFrame(predictionForKaggle, columns=['ID','Label'])
predictionForKaggle_df.to_csv("UnbVTPrediction.csv", index=False)