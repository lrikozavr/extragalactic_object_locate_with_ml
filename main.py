#!/home/kiril/python_env_iron_ment/new_proj/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import sys

from ml_network import NN
from data_processing import data_begin

save_path = ""
path_ml = ""
name_sample = "allofthis"

data = data_begin(save_path)

data = data.sample(frac=1, replace=True, random_state=1)

features = ['W1mag&W2mag', 'W1mag&W3mag', 'W1mag&phot_g_mean_mag', 'W1mag&phot_bp_mean_mag', 'W1mag&phot_rp_mean_mag', 
'W2mag&W3mag', 'W2mag&phot_g_mean_mag', 'W2mag&phot_bp_mean_mag', 'W2mag&phot_rp_mean_mag', 
'W3mag&phot_g_mean_mag', 'W3mag&phot_bp_mean_mag', 'W3mag&phot_rp_mean_mag', 
'phot_g_mean_mag&phot_bp_mean_mag', 'phot_g_mean_mag&phot_rp_mean_mag', 
'phot_bp_mean_mag&phot_rp_mean_mag']

batch_size = 1024
num_ep = 100
optimazer = 'adam'
loss = 'binary_crossentropy'

NN(data[features].values,data["Y"].values,0.4,0.4,batch_size,num_ep,optimazer,loss,
f"{path_ml}/predict_{name_sample}",
f"{path_ml}/model/mod_{name_sample}",
f"{path_ml}/model/weight_{name_sample}",
f"{path_ml}/{name_sample}")