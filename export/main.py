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
        self.name_main_sample = config['name_sample']
        self.name_sample = config['name_sample'] + config['additional_name']
        self.general_path = config['general_path']
        self.data_path = config['data_path']
        self.prediction_path = config['prediction_path']
        self.flags = config['flags']
        self.hyperparam = config['hyperparam']
        self.features = config['features']
        self.name_class = config['name_class']
        self.name_class_column = config["name_class_column"]
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

if not os.path.isdir(config.general_path):
    os.mkdir(config.general_path)


dir(config.general_path,'sample')
dir(config.general_path,'statistic')
dir(config.general_path,'ml')
dir(config.path_ml,'data')
dir(config.path_ml,'model')
dir(config.path_ml,'eval')
dir(config.path_ml,'prediction')
dir(config.path_ml,'picture')

#data download
if(config.flags["data_downloading"]["class_diff"]):
    diff_class(config)

if(config.flags["data_downloading"]["work"]):
    #stat_mass = []
    sum_mass = pd.DataFrame()

    for name in config.name_class:
        stat = class_download(name, config.path_sample,config)
        stat.to_csv(f'{config.path_stat}/{name}_slice.log', index=False)
        stat = pd.read_csv(f'{config.path_stat}/{name}_slice.log',header=0,sep=",")
        #print(np.array(stat.sum(axis=0)))
        sum = pd.DataFrame([np.array(stat.sum(axis=0))], columns = stat.columns.values, index = [name])
        #print(sum)
        sum_mass = pd.concat([sum_mass,sum], ignore_index=False)
        #stat_mass.append(stat)
    sum_mass.to_csv(f'{config.path_stat}/classes.log')
    print(sum_mass)


data = pd.DataFrame()
#data preparation
if(not config.flags['data_preprocessing']['work']):
    if(os.path.isfile(f'{config.path_ml_data}/{config.name_main_sample}_all.csv')):
        data = pd.read_csv(f'{config.path_ml_data}/{config.name_main_sample}_all.csv', header = 0, sep = ',')
    else:
        data = data_preparation(config.path_ml_data,config.path_sample,config.name_class,config)
else:
    data = data_preparation(config.path_ml_data,config.path_sample,config.name_class,config)    

#############################################################if(config.features["mod"])

#data_statistic
if(config.statistic["metric"]):
    data.describe().transpose().to_csv(f'{config.path_stat}/{config.name_main_sample}_stat.log')

from data_process import get_features
#network training
if(config.hyperparam["model_variable"]["work"]):
    #features from config
    #name from config
    print('Sample name: ', config.name_sample)
    print('Features: ', config.features["data"])
    #hyperparams from config
    batch_size = config.hyperparam['batch_size']
    num_ep = config.hyperparam['num_ep']
    optimizer = config.hyperparam['optimizer']
    loss = config.hyperparam['loss']
    validation_split = config.hyperparam['validation_split']
    #balanced class
    class_weights = None
    if(config.hyperparam["model_variable"]["balanced"]):
        from sklearn.utils import class_weight
        y = np.zeros(data.shape[0])
        cl = np.array(data[config.name_class_cls].values)
        for i in range(data.shape[0]):
            y[i] = np.argmax(cl[i,:])
        class_weights = dict(enumerate(class_weight.compute_class_weight(class_weight = 'balanced',classes = np.unique(y),y = y)))
        print("class weights",class_weights)
        del y

    print(data)

    print("Features mode list:\t",config.features["train"])
    features = get_features(config.features["train"],config)
    print("Features train values:\t",features)

    sample_weight = None
    if(config.hyperparam["model_variable"]["sample_weight"] in config.flags['data_preprocessing']['main_sample']['weight']['method']):
        #sample_weight = data[config.hyperparam["model_variable"]["sample_weight"]].values
        sample_weight = data[config.hyperparam["model_variable"]["sample_weight"]].values.T[0]
        print(sample_weight)

    try:
        data[features]
    except:
        raise Exception("data don't have initiated features, check config.features['train'] value and WARNINGs above")

    if(config.hyperparam["model_variable"]["work"]):
        NN(data[features],data[config.name_class_cls],data['z'],sample_weight,validation_split,batch_size,num_ep,optimizer,loss,class_weights,
        output_path_predict = config.path_predict,
        output_path_mod = config.path_model,
        output_path_weight = config.path_weight,
        path_save_eval = config.path_eval,
        config=config)

#statistic
from statistic import metric_statistic
if(config.statistic["metric"]):
    metric_statistic(config)

print(data)
#picture
if(config.picture["work"]):
    from graphic import picture_confusion_matrix, picture_roc_prc, picture_hist, TSNE_pic, contam_dist_pic, multigridplot, picture_correlation_matrix
    
    if(config.picture["correlation_matrix"]):
        picture_correlation_matrix(data[get_features(config.features["train"],config)],"mags",config)
    
    if(config.picture["tSNE"]["work"]):
        TSNE_pic(data,config)

    if(config.picture["contam_dist"]["work"]):
        contam_dist_pic(data,config)

    if(config.picture["multigridplot"]):
        multigridplot(data,get_features(config.features["train"],config),config)

    if(config.picture["roc_prc"]["work"]):
        picture_roc_prc(config)
    #to network   
    if(config.picture["confusion_matrix"]):
        picture_confusion_matrix(config)
    if(config.picture["hist"]["work"]):
        picture_hist(config)

    from graphic import redshift_estimation
    if(config.hyperparam["redshift"]["picture"]):
        redshift_estimation(config)

#prediction
if(config.flags["prediction"]["work"]):
    from network import large_file_prediction
    large_file_prediction(config)

#short statistic
