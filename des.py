# -*- coding: utf-8 -*-

import os
from ml_network import LoadModel
from DataTransform import DataP
import pandas as pd
import numpy as np

slice_path = '/home/lrikozavr/ML_work/des_pro/ml/prediction'
#slice_path = "/home/lrikozavr/catalogs/des/slice"

output_path_predict = "/home/lrikozavr/ML_work/des_pro/ml/prediction_sm"

inform_path = '/home/lrikozavr/ML_work/des_pro/ml/model'

optimizer = 'adam'
loss = 'binary_crossentropy'
batch_size = 1024

def local_ML(output_path_predict,data,train,batch_size,output_path_mod,output_path_weight,optimizer,loss,name):
    model = LoadModel(output_path_mod,output_path_weight,optimizer,loss)
    Class = model.predict(DataP(train,0), batch_size)
    
    #Class = np.array(Class)
    #data[f'{name}_probability'] = Class
    res = pd.DataFrame(np.array(Class), columns=["star_cls_prob","qso_cls_prob","gal_cls_prob"])
    data = pd.concat([data,res], axis=1)
    data.to_csv(output_path_predict, index=False)

index=0
count = len(os.listdir(slice_path))

output_path_mod = f"{inform_path}/mod_extragal_custom_sm_1n15n11n7"
output_path_weight = f"{inform_path}/weight_extragal_custom_sm_1n15n11n7"

#output_path_mod = f"{inform_path}/mod_extragal_custom_1n9n8n4"
#output_path_weight = f"{inform_path}/weight_extragal_custom_1n9n8n4"

flag=0
for name in os.listdir(slice_path):
    #if((not name=="file_12") and (not name=="file_4") and (not name=="file_00")):
    if (name == 'file_16.csv'):
        print(flag)
        flag = 1
    #print(os.listdir(slice_path))
    if(flag):
        index += 1
        file_path = f"{slice_path}/{name}"
        print(file_path)
    #"TTYPE8" "TTYPE9" 						EBV_SFD98 	  23  
    #										MagAutoDered  24 25 26 27 28
    #												Flags 36 39 42 45 48  	
    #										MagAuto		  151 152 153 154 155
    #										MagAutoErr	  156 157 158 159 160
    #										ClassStar	  191 192 193 194 195
        data = pd.read_csv(file_path, header=0, sep=',')
        #data = pd.read_csv(file_path, header=None, sep=',')
        #data.columns = ['RA','DEC','EBV_SFD98','g_MagAD','r_MagAD','i_MagAD','z_MagAD','Y_MagAD','g_Flag','r_Flag','i_Flag','z_Flag','Y_Flag','g_MagA','r_MagA','i_MagA','z_MagA','Y_MagA','g_MagAE','r_MagAE','i_MagAE','z_MagAE','Y_MagAE','g_clS','r_clS','i_clS','z_clS','Y_clS']
        #train = data.drop(['RA','DEC','EBV_SFD98','g_MagAD','r_MagAD','i_MagAD','z_MagAD','Y_MagAD','g_Flag','r_Flag','i_Flag','z_Flag','Y_Flag','g_MagAE','r_MagAE','i_MagAE','z_MagAE','Y_MagAE','g_clS','r_clS','i_clS','z_clS','Y_clS'], axis=1)
        train = data.drop(['RA','DEC','EBV_SFD98','g_MagA','r_MagA','i_MagA','z_MagA','Y_MagA','g_Flag','r_Flag','i_Flag','z_Flag','Y_Flag','g_MagAE','r_MagAE','i_MagAE','z_MagAE','Y_MagAE','g_clS','r_clS','i_clS','z_clS','Y_clS'], axis=1)
        features=['g_MagAD','r_MagAD','i_MagAD','z_MagAD','Y_MagAD']
        local_ML(f"{output_path_predict}/{name}.csv",data,train[features],batch_size,output_path_mod,output_path_weight,optimizer,loss,'exgal')
        print(f"Status: {index/float(count) *100}")
        del train
        del data