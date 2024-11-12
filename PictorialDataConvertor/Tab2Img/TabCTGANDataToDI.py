# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 12:32:53 2024

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

CTGAN_data = pd.read_csv("../../Data/TabData/CTGANSynthData1.csv").to_numpy()

model = Tab2Img()
model.fit(training_data, training_labels)
images = model.transform(CTGAN_data)

for i in range(len(images)):
    image = Image.fromarray(images[i]).convert('I').resize((16,16))
    image.save("../../Data/PictData/DeepInsight/CTGANMalignSynthData/sample"+str(i+1)+".png", 'PNG')
    