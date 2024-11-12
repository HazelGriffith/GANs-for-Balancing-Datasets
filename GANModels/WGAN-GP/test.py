# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 22:43:08 2024

@author: hazel
"""

import pandas as pd
from ydata_synthetic.synthesizers.regular import RegularSynthesizer
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters

#Load data and define the data processor parameters
data = pd.read_csv("../../Data/TabData/Train.csv")
malignData = data[data['Label'] == 1]
malignData = malignData.drop(labels='Label', axis=1)
num_cols = malignData.columns.tolist()
cat_cols = []

#Defining the training parameters
noise_dim = 625
dim = 56
batch_size = 32

log_step = 100
epochs = 500+1
learning_rate = [5e-4, 3e-3]
beta_1 = 0.5
beta_2 = 0.9
models_dir = '../cache'

gan_args = ModelParameters(batch_size=batch_size,
                           lr=learning_rate,
                           betas=(beta_1, beta_2),
                           noise_dim=noise_dim,
                           layers_dim=dim)

train_args = TrainParameters(epochs=epochs,
                             sample_interval=log_step)

synth = RegularSynthesizer(modelname='wgangp', model_parameters=gan_args, n_critic=2)
synth.fit(malignData, train_args, num_cols, cat_cols)

synth.save('tabWGANGP1.pkl')

#########################################################
#    Loading and sampling from a trained synthesizer    #
#########################################################
synth = RegularSynthesizer.load('tabWGANGP1.pkl')
synth_data = synth.sample(1535)
synth_data.to_csv("../../Data/Tabdata/WGANGPSynthData1.csv")