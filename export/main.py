# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import sys
import argparse

from network import NN
from data_process import data_preparation
from data_download import class_download, diff_class

#argument parse
def parse_arg():
    parser = argparse.ArgumentParser(
        description = '',
        epilog = '@lrikozavr'
    )
    parser.add_argument(
        '--config','-c', dest = 'config',
        type = str, help = '''
        json format
        '''
    )
    return parser.parse_args()
args = parse_arg()

#general_path = '/home/lrikozavr/ML_work/allwise_gaiadr3'

import json
fconfig = open(args.config)
def parse_config(fconfig):
    config = json.load(fconfig)
    name_sample = config['name_sample']
    general_path = config['general_path']
    data_path = config['data_path']
    prediction_path = config['prediction_path']
    flags = config['flags']
    hyperparam = config['hyperparam']
    features = config['features']
    name_class = config['name_class']
    return name_sample, general_path, data_path, prediction_path, flags, hyperparam, features, name_class

name_sample, general_path, data_path, prediction_path, flags, hyperparam, features, name_class = parse_config(fconfig)
fconfig.close()

def dir(save_path,name):
    dir_name = f"{save_path}/{name}"
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

path_data = f'{general_path}/ml/data'
path_ml = f'{general_path}/ml'
path_sample = f'{general_path}/sample'
path_pic = f'{general_path}/ml/picture'

dir(general_path,'ml')
dir(path_ml,'data')
dir(path_ml,'model')
dir(path_ml,'eval')
dir(path_ml,'prediction')
dir(path_ml,'picture')

#data download
diff_class(data_path,name_class,path_sample)
for name in name_class:
    class_download(name, path_sample)


data = pd.DataFrame()
#data preparation
if(not flags['data_preprocessing']['work']):
    if(os.path.isfile(f'{path_data}/all.csv')):
        data = pd.read_csv(f'{path_data}/all.csv', header = 0, sep = ',')
    else:
        data = data_preparation(path_data,path_sample,name_class)
else:
    data = data_preparation(path_data,path_sample,name_class)    

#network training
#features from config
#name from config
print('Sample name: ', name_sample)
print('Features: ', features)
#hyperparams from config
batch_size = hyperparam['batch_size']
num_ep = hyperparam['num_ep']
optimazer = hyperparam['optimizer']
loss = hyperparam['loss']
validation_split = hyperparam['validation_split']
#balanced class
from sklearn.utils import class_weight
class_weights = {}

print(data)
NN(data[features].values,data[name_class].values,validation_split,batch_size,num_ep,optimazer,loss,class_weights,
f"{path_ml}/prediction/{name_sample}",
f"{path_ml}/model/mod_{name_sample}",
f"{path_ml}/model/weight_{name_sample}",
f"{path_ml}/eval/{name_sample}")

#statistic
#picture
#prediction
#short statistic
