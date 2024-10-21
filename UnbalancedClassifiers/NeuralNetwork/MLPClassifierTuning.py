# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 10:51:04 2024

@author: hazel
"""
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

training_data = pd.read_csv("../../Data/TabData/Train.csv")
training_labels = training_data['Label'].to_numpy()
training_data = training_data.drop(labels='Label',axis=1).to_numpy()

mlp = MLPClassifier(activation='relu',solver='adam',learning_rate='adaptive',random_state=42,early_stopping=True)

param_grid = [{'hidden_layer_sizes':([70,17],[71,16],[72,15],[73,14],[74,13],[69,18],[68,19],[67,20],[66,21],[65,22]),
               'learning_rate_init':(1e-3,1e-1,1e-2)
             }]

gscv = GridSearchCV(mlp, param_grid, n_jobs=4, scoring='neg_root_mean_squared_error')
gscv.fit(training_data,training_labels)
print(gscv.best_params_)
print(gscv.best_score_)
