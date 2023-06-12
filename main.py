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

general_path='/home/lrikozavr/ML_work/allwise_gaiadr3'
#general_path='/home/lrikozavr/ML_work/des_pro'
#general_path='/home/lrikozavr/ML_work/des_z'
save_path = f'{general_path}/ml/data'
#path_ml = "/media/kiril/j_08/ML/extragal"
path_ml = f'{general_path}/ml'
path_sample = f'{general_path}/sample'
#path_sample = f'{general_path}/sample_3'

dir(general_path,'ml')
dir(path_ml,'data')
dir(path_ml,'model')
dir(path_ml,'eval')
dir(path_ml,'prediction')

#data_1_0, data_0_1 = data_begin(save_path,path_sample)
#data = data_begin(save_path,path_sample)
data = pd.read_csv(f'{save_path}/all.csv', header = 0, sep = ',')

def body(data, name_sample, features):
    print(name_sample)
    import matplotlib.pyplot as plt

    f = plt.figure(figsize=(25, 21))
    plt.matshow(data[features].corr(),fignum=f.number)
    plt.xticks(range(data[features].shape[1]),data[features].columns.values, fontsize=14, rotation=90)
    plt.yticks(range(data[features].shape[1]),data[features].columns.values, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16)
    plt.savefig('/home/lrikozavr/ML_work/allwise_gaiadr3/ml/data/w123_phot.png')
    #plt.show()
    
    exit()
    #print(data['Y'])

    '''
    features = ['gmag&rmag', 'gmag&imag', 'gmag&zmag', 'gmag&Ymag', 
    'rmag&imag', 'rmag&zmag', 'rmag&Ymag', 
    'imag&zmag', 'imag&Ymag', 
    'zmag&Ymag',
    'gmag','rmag','imag','zmag','Ymag']
    '''
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
    
    print(data)
    NN(data[features].values,data[["star_cls","qso_cls","gal_cls"]].values,0.3,0.3,batch_size,num_ep,optimazer,loss,
    f"{path_ml}/prediction/{name_sample}",
    f"{path_ml}/model/mod_{name_sample}",
    f"{path_ml}/model/weight_{name_sample}",
    f"{path_ml}/eval/{name_sample}")

#body(data_1_0,'extragal')
features = ['W1mag&W2mag', 'W1mag&W3mag', 'W1mag&phot_g_mean_mag', 'W1mag&phot_bp_mean_mag', 'W1mag&phot_rp_mean_mag', 
    'W2mag&W3mag', 'W2mag&phot_g_mean_mag', 'W2mag&phot_bp_mean_mag', 'W2mag&phot_rp_mean_mag', 
    'W3mag&phot_g_mean_mag', 'W3mag&phot_bp_mean_mag', 'W3mag&phot_rp_mean_mag', 
    'phot_g_mean_mag&phot_bp_mean_mag', 'phot_g_mean_mag&phot_rp_mean_mag', 
    'phot_bp_mean_mag&phot_rp_mean_mag',
    'W1mag','W2mag','W3mag','phot_g_mean_mag','phot_bp_mean_mag','phot_rp_mean_mag']
body(data,'qso_gal_star_w123',features)
features = ['W1mag&W2mag', 'W1mag&phot_g_mean_mag', 'W1mag&phot_bp_mean_mag', 'W1mag&phot_rp_mean_mag', 
    'W2mag&phot_g_mean_mag', 'W2mag&phot_bp_mean_mag', 'W2mag&phot_rp_mean_mag',  
    'phot_g_mean_mag&phot_bp_mean_mag', 'phot_g_mean_mag&phot_rp_mean_mag', 
    'phot_bp_mean_mag&phot_rp_mean_mag']
body(data,'qso_gal_star_w12',features)

