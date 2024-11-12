# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 11:27:07 2024

@author: hazel
"""

import os
import shutil
import numpy as np

labels = np.loadtxt("trainingLabels.csv", delimiter=',').astype(np.float32)

m = 1
b = 1
for num, label in enumerate(labels, start=1):
    
    if label == 1:
        shutil.move("DeepInsight/sample"+str(num)+".png", "DeepInsight/malign/sample"+str(m)+".png")
        m+=1
    else:
        shutil.move("DeepInsight/sample"+str(num)+".png", "DeepInsight/benign/sample"+str(b)+".png")
        b+=1