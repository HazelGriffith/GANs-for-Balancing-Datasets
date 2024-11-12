# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 16:33:05 2024

@author: hazel
"""

from sklearn.datasets import fetch_covtype
from tab2img.converter import Tab2Img
import numpy as np
import pandas as pd
from PIL import Image

loaded_data = pd.read_csv("../../Data/TabData/Validation.csv")
training_labels = loaded_data['ID'].tolist()
training_data = loaded_data.drop(labels='ID', axis=1).to_numpy()

model = Tab2Img()
images = model.fit_transform(training_data, training_labels)
for i in range(len(images)):
    image = Image.fromarray(images[i]).convert('I').resize((16,16))
    image.save("../../Data/PictData/DeepInsight/Validation/sample"+str(i+1)+".png", 'PNG')
