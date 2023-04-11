# -*- coding: utf-8 -*-

import os
from ml_network import LoadModel
from DataTransform import DataP
import pandas as pd
import numpy as np

#slice_path = "/home/kiril/github/ML_data/gaia_all_cat"
slice_path = "/media/kiril/j_08/CATALOGUE/gaia_all_cat/long"
#output_path_mod_one = "/home/kiril/github/ML_with_AGN/ML/models/mod_one_AGN_STAR_GALAXY_QSO"
#output_path_weight_one = "/home/kiril/github/ML_with_AGN/ML/models/weight_one_AGN_STAR_GALAXY_QSO"

#output_path_mod_dark = "/home/kiril/github/ML_with_AGN/ML/models/mod_dark_STAR_AGN_GALAXY_QSO"
#output_path_weight_dark = "/home/kiril/github/ML_with_AGN/ML/models/weight_dark_STAR_AGN_GALAXY_QSO"


#output_path_predict = "/media/kiril/j_08/AGN/predict/Gaia_AllWISE"
output_path_predict = "/media/kiril/j_08/ML/extragal/predict/Gaia_AllWISE"

name_sample = list(['gal','qso'])
inform_path = "/media/kiril/j_08/ML/extragal"


optimizer = 'adam'
loss = 'binary_crossentropy'
batch_size = 1024

def local_ML(output_path_predict,data,train,batch_size,output_path_mod,output_path_weight,optimizer,loss,name):
    model = LoadModel(output_path_mod,output_path_weight,optimizer,loss)
    Class = model.predict(DataP(train,1), batch_size)

    Class = np.array(Class)
    data[f'{name}_probability'] = Class
    data.to_csv(output_path_predict, index=False)
index=0
count = len(os.listdir(slice_path))
for name_path in name_sample:
    output_path_mod_one = f"{inform_path}/model/mod_{name_path}_custom_1"
    output_path_weight_one = f"{inform_path}/model/weight_{name_path}_custom_1"
    output_path_mod_dark = f"{inform_path}/model/mod_{name_path}_linear_1"
    output_path_weight_dark = f"{inform_path}/model/weight_{name_path}_linear_1"
    
    for name in os.listdir(slice_path):
        #if((not name=="file_12") and (not name=="file_4") and (not name=="file_00")):
        if(not name=="file_"):
            index += 1
            file_path = f"{slice_path}/{name}"
            print(file_path)
        
            data = pd.read_csv(file_path, header=0, sep=',')
            data.columns = ['RA','DEC','eRA','eDEC','plx','eplx','pmra','pmdec','epmra','epmdec','ruwe','g','bp','rp','RAw','DECw','w1','ew1','snrw1','w2','ew2','snrw2','w3','ew3','snrw3','w4','ew4','snrw4','dra','ddec']
            train = data.drop(['RA','DEC','eRA','eDEC','plx','eplx','pmra','pmdec','epmra','epmdec','ruwe'], axis=1)
            train = train.drop(['RAw','DECw','ew1','snrw1','ew2','snrw2','ew3','snrw3','w4','ew4','snrw4','dra','ddec'], axis=1)
            features=['w1','w2','w3','g','bp','rp']
            local_ML(f"{output_path_predict}_{name}_{name_path}_normal.csv",data,train[features],batch_size,output_path_mod_one,output_path_weight_one,optimizer,loss,name_path)
            local_ML(f"{output_path_predict}_{name}_{name_path}_dark.csv",data,train[features],batch_size,output_path_mod_dark,output_path_weight_dark,optimizer,loss,name_path)
            print(f"Status: {index/float(count) *100}")
            del train
            del data