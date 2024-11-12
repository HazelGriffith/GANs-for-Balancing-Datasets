# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 12:08:45 2024

@author: hazel
"""

import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import Metadata

data = pd.read_csv("../../Data/TabData/Train.csv")
malignData = data[data['Label'] == 1]
malignData = malignData.drop(labels='Label', axis=1)

metadata = Metadata.detect_from_dataframe(data=malignData, table_name='malicious_data')

synthesizer = CTGANSynthesizer(metadata, verbose=True)

synthesizer.fit(malignData)
fig = synthesizer.get_loss_values_plot()
fig.show()
synthesizer.save("CTGANmodel.pkl")

synthetic_data = synthesizer.sample(num_rows=1535)
synthetic_data.to_csv("../../Data/Tabdata/CTGANSynthData1.csv")