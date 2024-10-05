# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 13:32:18 2024

@author: hazel
"""

from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd

training_data = pd.read_csv("../../Data/TabData/Train.csv")
training_labels = training_data['Label'].to_numpy()
training_data = training_data.drop(labels='Label',axis=1).to_numpy()
validation_data = pd.read_csv("../../Data/TabData/Validation.csv")
validation_ids = np.expand_dims(validation_data['ID'].to_numpy(), axis=1)
validation_data = validation_data.drop(labels='ID',axis=1).to_numpy()

GNBC = GaussianNB()
GNBC.fit(training_data, training_labels)
prediction = np.expand_dims(GNBC.predict(validation_data), axis=1)

predictionForKaggle = np.concatenate((validation_ids,prediction), axis=1)
predictionForKaggle_df = pd.DataFrame(predictionForKaggle, columns=['ID','Label'])
predictionForKaggle_df.to_csv("UnbNBPrediction.csv", index=False)