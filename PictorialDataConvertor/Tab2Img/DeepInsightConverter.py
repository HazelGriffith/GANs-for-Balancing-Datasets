# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 16:33:05 2024

@author: hazel
"""

from sklearn.datasets import fetch_covtype
from tab2img.converter import Tab2Img
import numpy as np
import pandas as pd

loaded_data = pd.read_csv("../../Data/TabData/Train.csv")
training_labels = loaded_data['Label'].tolist()
training_data = loaded_data.drop(labels='Label', axis=1).to_numpy()

model = Tab2Img()
images = model.fit_transform(training_data, training_labels).reshape(4465,144)
np.savetxt("../../Data/PictData/DITrainingData.csv", images, delimiter=',')
np.savetxt("../../Data/trainingLabels.csv", training_labels, delimiter=',')