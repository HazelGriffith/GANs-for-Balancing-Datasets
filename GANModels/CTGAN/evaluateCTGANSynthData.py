# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 12:41:05 2024

@author: hazel
"""

import pandas as pd
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality
from sdv.metadata import Metadata

real_data = pd.read_csv("../../Data/TabData/Train.csv")
malignData = real_data[real_data['Label'] == 1]
malignData = malignData.drop(labels='Label', axis=1)
synth_data = pd.read_csv("../../Data/TabData/CTGANSynthData1.csv")
synth_data = synth_data.drop(synth_data.columns[0],axis=1)

metadata = Metadata.detect_from_dataframe(data=malignData, table_name='malicious_data')


diagnostic_report = run_diagnostic(malignData, synth_data, metadata)

quality_report = evaluate_quality(malignData, synth_data, metadata)