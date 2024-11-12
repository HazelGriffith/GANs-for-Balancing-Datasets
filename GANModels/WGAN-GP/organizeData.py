# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 12:30:29 2024

@author: hazel
"""

import pandas as pd

data = pd.read_csv("../../Data/TabData/Train.csv")
malignData = data[data['Label'] == 1]
print(malignData)