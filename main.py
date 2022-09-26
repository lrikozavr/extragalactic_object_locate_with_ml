#!/home/kiril/python_env_iron_ment/new_proj/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import sys

from ml_network import NN
from data_processing import data_begin

#save_path = "/media/kiril/j_08/ML/extragal/"
save_path = 'ml/prediction'
#path_ml = "/media/kiril/j_08/ML/extragal"
path_ml = 'ml'

path_sample = 'ml'

name_sample = "allofthis"

#data_1_0_0, data_0_1_0, data_0_0_1 = data_begin(save_path)

data_1_0, data_0_1 = data_begin(save_path,path_sample)

def body(data, name_sample):
    print(name_sample)
    
    #data = data.sample(frac=1, replace=True, random_state=1)
    #data = data.reset_index(drop=True)
    #print(data['Y'])
    #data.fillna(0)
    '''
    features = ['W1mag&W2mag', 'W1mag&W3mag', 'W1mag&phot_g_mean_mag', 'W1mag&phot_bp_mean_mag', 'W1mag&phot_rp_mean_mag', 
    'W2mag&W3mag', 'W2mag&phot_g_mean_mag', 'W2mag&phot_bp_mean_mag', 'W2mag&phot_rp_mean_mag', 
    'W3mag&phot_g_mean_mag', 'W3mag&phot_bp_mean_mag', 'W3mag&phot_rp_mean_mag', 
    'phot_g_mean_mag&phot_bp_mean_mag', 'phot_g_mean_mag&phot_rp_mean_mag', 
    'phot_bp_mean_mag&phot_rp_mean_mag']
    '''
    features = ['W1mag&W2mag', 'W1mag&phot_g_mean_mag', 'W1mag&phot_bp_mean_mag', 'W1mag&phot_rp_mean_mag', 
    'W2mag&phot_g_mean_mag', 'W2mag&phot_bp_mean_mag', 'W2mag&phot_rp_mean_mag',  
    'phot_g_mean_mag&phot_bp_mean_mag', 'phot_g_mean_mag&phot_rp_mean_mag', 
    'phot_bp_mean_mag&phot_rp_mean_mag']

    batch_size = 1024
    num_ep = 10
    optimazer = 'adam'
    loss = 'binary_crossentropy'

    NN(data[features].values,data["Y"].values,0.3,0.3,batch_size,num_ep,optimazer,loss,
    f"{path_ml}/prediction/{name_sample}",
    f"{path_ml}/model/mod_{name_sample}",
    f"{path_ml}/model/weight_{name_sample}",
    f"{path_ml}/eval/{name_sample}")

#body(data_1_0_0,"gal")
#body(data_0_1_0,"qso")
#body(data_0_0_1,"star")

body(data_1_0,'extragal')