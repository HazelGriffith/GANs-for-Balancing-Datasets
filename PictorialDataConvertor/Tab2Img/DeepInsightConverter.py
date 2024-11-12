# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 16:33:05 2024

@author: hazel
"""

import os
import shutil
from sklearn.datasets import fetch_covtype
from tab2img.converter import Tab2Img
import numpy as np
import pandas as pd
from PIL import Image

loaded_data = pd.read_csv("../../Data/TabData/Train.csv")
training_labels = loaded_data['Label'].tolist()
training_data = loaded_data.drop(labels='Label', axis=1).to_numpy()

validation_data = pd.read_csv("../../Data/TabData/Validation.csv")
validation_data = validation_data.drop(labels='ID', axis=1).to_numpy()

model = Tab2Img()
images = model.fit_transform(training_data, training_labels)
m = 1
b = 1
for num, label in enumerate(training_labels, start=1):
    image = Image.fromarray(images[num-1]).convert('I').resize((16,16))
    image.save("../../Data/PictData/DeepInsight/sample"+str(num)+".png", 'PNG')
    if label == 1:
        shutil.move("../../Data/PictData/DeepInsight/sample"+str(num)+".png", "../../Data/PictData/DeepInsight/malign/sample"+str(m)+".png")
        m+=1
    else:
        shutil.move("../../Data/PictData/DeepInsight/sample"+str(num)+".png", "../../Data/PictData/DeepInsight/benign/sample"+str(b)+".png")
        b+=1
        
images = model.transform(validation_data)

for i in range(len(images)):
    image = Image.fromarray(images[i]).convert('I').resize((16,16))
    image.save("../../Data/PictData/DeepInsight/Validation/sample"+str(i+1)+".png", 'PNG')
    