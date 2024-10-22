"""
Created on Mon Oct 21 2024

@author: Puxin
@description: This script processes tabular data and transforms it into images using the IGTD method.
"""

from TINTOlib.igtd import IGTD
import pandas as pd

df = pd.read_csv("../../Data/TabData/Train.csv")

model = IGTD(scale=[12,12]) 

model.generateImages(df, 'TestImages')
