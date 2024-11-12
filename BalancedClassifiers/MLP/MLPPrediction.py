# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 14:19:19 2024

@author: hazel
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

training_data = pd.read_csv("../../Data/TabData/Train.csv")
training_labels = training_data['Label'].to_numpy()
training_data = training_data.drop(labels='Label',axis=1).to_numpy()

synth_data = pd.read_csv("../../Data/TabData/CTGANSynthData1.csv").to_numpy()
training_data = np.concatenate((training_data, synth_data),axis=0)
training_labels = np.concatenate((training_labels, np.ones(1535, dtype=int)),axis=0)

validation_data = pd.read_csv("../../Data/TabData/Validation.csv")
validation_ids = np.expand_dims(validation_data['ID'].to_numpy(), axis=1)
validation_data = validation_data.drop(labels='ID',axis=1).to_numpy()

mlp = MLPClassifier(hidden_layer_sizes=[65,22],activation='relu',solver='adam',learning_rate='adaptive',learning_rate_init=1e-2,random_state=42,early_stopping=True)

mlp.fit(training_data, training_labels)
prediction = np.expand_dims(mlp.predict(validation_data), axis=1)

predictionForKaggle = np.concatenate((validation_ids,prediction), axis=1)
predictionForKaggle_df = pd.DataFrame(predictionForKaggle, columns=['ID','Label'])
predictionForKaggle_df.to_csv("BalMLPPrediction.csv", index=False)