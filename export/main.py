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
class Config():

    def __init__(self,fconfig):
        config = json.load(fconfig)
        self.name_sample = config['name_sample']
        self.general_path = config['general_path']
        self.data_path = config['data_path']
        self.prediction_path = config['prediction_path']
        self.flags = config['flags']
        self.hyperparam = config['hyperparam']
        self.features = config['features']
        self.name_class = config['name_class']
        self.base = config["base"]
        self.picture = config['picture']
        self.statistic = config['statistic']
        #
        self.path_ml_data = f'{self.general_path}/ml/data'
        self.path_ml = f'{self.general_path}/ml'
        self.path_sample = f'{self.general_path}/sample'
        self.path_pic = f'{self.general_path}/ml/picture'
        self.path_stat = f'{self.general_path}/statistic'
        #
        self.path_model = f"{self.path_ml}/model/mod_{self.name_sample}"
        self.path_weight = f"{self.path_ml}/model/weight_{self.name_sample}"
        self.path_eval = f"{self.path_ml}/eval/{self.name_sample}"
        self.path_predict = f"{self.path_ml}/prediction/{self.name_sample}"
        #
        columns = []
        columns_prob = []
        for col in self.name_class:
            columns.append(f"{col}_cls")
            columns_prob.append(f"{col}_cls_prob")
        self.name_class_cls = columns
        self.name_class_prob = columns_prob






#    return name_sample, general_path, data_path, prediction_path, flags, hyperparam, features, name_class

#name_sample, general_path, data_path, prediction_path, flags, hyperparam, features, name_class = parse_config(fconfig)
config = Config(fconfig)
fconfig.close()

def dir(save_path,name):
    dir_name = f"{save_path}/{name}"
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

if not os.path.isdir(config.prediction_path):
    os.mkdir(config.prediction_path)


dir(config.general_path,'sample')
dir(config.general_path,'statistic')
dir(config.general_path,'ml')
dir(config.path_ml,'data')
dir(config.path_ml,'model')
dir(config.path_ml,'eval')
dir(config.path_ml,'prediction')
dir(config.path_ml,'picture')

#data download
#stat_mass = []
sum_mass = pd.DataFrame()
diff_class(config.data_path,config.name_class,config.path_sample,config.base)
for name in config.name_class:
    stat = class_download(name, config.path_sample,config)
    stat.to_csv(f'{config.path_stat}/{name}_slice.log', index=False)
    sum = pd.DataFrame(stat.sum(axis=0), columns = stat.columns.values, index = name)
    sum_mass = pd.concat([sum_mass,sum], ignore_index=False)
    #stat_mass.append(stat)
sum_mass.to_csv(f'{config.path_stat}/classes.log')
print(sum_mass)


data = pd.DataFrame()
#data preparation
if(not config.flags['data_preprocessing']['work']):
    if(os.path.isfile(f'{config.path_ml_data}/all.csv')):
        data = pd.read_csv(f'{config.path_ml_data}/all.csv', header = 0, sep = ',')
    else:
        data = data_preparation(config.path_ml_data,config.path_sample,config.name_class,config)
else:
    data = data_preparation(config.path_ml_data,config.path_sample,config.name_class,config)    

#data_statistic
data.describe().to_csv(f'{config.path_stat}/stat.log')

#network training
#features from config
#name from config
print('Sample name: ', config.name_sample)
print('Features: ', config.features)
#hyperparams from config
batch_size = config.hyperparam['batch_size']
num_ep = config.hyperparam['num_ep']
optimizer = config.hyperparam['optimizer']
loss = config.hyperparam['loss']
validation_split = config.hyperparam['validation_split']
#balanced class
from sklearn.utils import class_weight
class_weights = {}

print(data)


#


from data_process import get_features
features = get_features(config.features["train"],config)

if(config.hyperparam["model_variable"]["work"]):
    NN(data[features].values,data[config.name_class].values,validation_split,batch_size,num_ep,optimizer,loss,class_weights,
    output_path_predict = config.path_predict,
    output_path_mod = config.path_model,
    output_path_weight = config.path_weight,
    path_save_eval = config.path_eval,
    config=config)

#statistic
from statistic import metric_statistic
if(config.statistic["metric"]):
    metric_statistic(config)

#picture
from graphic import picture_cm, picture_loss, picture_roc_prc, picture_hist, picture_metrics
if(not 0 in config.picture["roc_prc"]):
    picture_roc_prc(config)
if(config.picture["loss"]):
    picture_loss(optimizer,loss,config)
if(config.picture["cm"]):
    picture_cm(config)
if(config.picture["hist"]["work"]):
    picture_hist(config)
if(config.picture["metrics_h"]):
    picture_metrics(config)

#prediction

#short statistic
