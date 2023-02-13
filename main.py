#!/home/kiril/python_env_iron_ment/new_proj/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import sys

from ml_network import NN
from data_processing import data_begin

def dir(save_path,name):
    dir_name = f"{save_path}/{name}"
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
#save_path = "/media/kiril/j_08/ML/extragal/"

general_path='/home/lrikozavr/ML_work/des_pro'
#general_path='/home/lrikozavr/ML_work/des_z'
save_path = f'{general_path}/ml/data'
#path_ml = "/media/kiril/j_08/ML/extragal"
path_ml = f'{general_path}/ml'
path_sample = f'{general_path}/sample'

dir(general_path,'ml')
dir(path_ml,'data')
dir(path_ml,'model')
dir(path_ml,'eval')
dir(path_ml,'prediction')

data_1_0, data_0_1 = data_begin(save_path,path_sample)


def body(data, name_sample):
    print(name_sample)
    
    #print(data['Y'])
    '''
    features = ['W1mag&W2mag', 'W1mag&W3mag', 'W1mag&phot_g_mean_mag', 'W1mag&phot_bp_mean_mag', 'W1mag&phot_rp_mean_mag', 
    'W2mag&W3mag', 'W2mag&phot_g_mean_mag', 'W2mag&phot_bp_mean_mag', 'W2mag&phot_rp_mean_mag', 
    'W3mag&phot_g_mean_mag', 'W3mag&phot_bp_mean_mag', 'W3mag&phot_rp_mean_mag', 
    'phot_g_mean_mag&phot_bp_mean_mag', 'phot_g_mean_mag&phot_rp_mean_mag', 
    'phot_bp_mean_mag&phot_rp_mean_mag']
    '''
    features = ['gmag&rmag', 'gmag&imag', 'gmag&zmag', 'gmag&Ymag', 
    'rmag&imag', 'rmag&zmag', 'rmag&Ymag', 
    'imag&zmag', 'imag&Ymag', 
    'zmag&Ymag',
    'gmag','rmag','imag','zmag','Ymag']
#['e_gmag&e_rmag', 'e_gmag&e_imag', 'e_gmag&e_zmag', 'e_gmag&e_Ymag', 'e_rmag&e_imag', 'e_rmag&e_zmag', 'e_rmag&e_Ymag', 'e_imag&e_zmag', 'e_imag&e_Ymag', 'e_zmag&e_Ymag']
    
    '''
    features = ['W1mag&W2mag', 'W1mag&phot_g_mean_mag', 'W1mag&phot_bp_mean_mag', 'W1mag&phot_rp_mean_mag', 
    'W2mag&phot_g_mean_mag', 'W2mag&phot_bp_mean_mag', 'W2mag&phot_rp_mean_mag',  
    'phot_g_mean_mag&phot_bp_mean_mag', 'phot_g_mean_mag&phot_rp_mean_mag', 
    'phot_bp_mean_mag&phot_rp_mean_mag']
    '''
    batch_size = 1024
    num_ep = 20
    optimazer = 'adam'
    loss = 'binary_crossentropy'

    NN(data[features].values,data["Y"].values,0.3,0.3,batch_size,num_ep,optimazer,loss,
    f"{path_ml}/prediction/{name_sample}",
    f"{path_ml}/model/mod_{name_sample}",
    f"{path_ml}/model/weight_{name_sample}",
    f"{path_ml}/eval/{name_sample}")

body(data_1_0,'extragal')