# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os

from fuzzy_options import fuzzy_dist,fuzzy_err, Normali, colors, MCD, redded_des

def NtoPtoN(data,index):
    res = []
    for i in index:
        res.append(data.iloc[i])
    res = pd.DataFrame(np.array(res), columns=data.columns.values)
    return res

def process(data,name,save_path):
    #data_mags = data.drop(['RA','DEC','z','CatName','Class'], axis=1)
    
    #redded_des(data)
    print(name, 'deredded complite')
    data_mags = data.drop(['RA','DEC','z'], axis=1)
    data_dist, data_err = colors(data_mags)
    print(name," complite colors")
    mcd_d, gauss_d, outlire = MCD(data_dist,0)
    print(name," complite MCD")

    #data = data.drop(index=outlire)
    #data_dist = data_dist.drop(index=outlire)
    #data_err = data_err.drop(index = outlire)

    mcd_g = pd.DataFrame(np.array(gauss_d), columns = ['mcd_g'])
    mcd_d = pd.DataFrame(np.array(mcd_d), columns = ['mcd_d'])
    #data = pd.concat([data[['RA','DEC','z','CatName','Class']],data_dist,data_err], axis=1)
    data = pd.concat([data[['RA','DEC','z','W1mag','W2mag','W3mag','phot_g_mean_mag','phot_bp_mean_mag','phot_rp_mean_mag']],data_dist,data_err,mcd_d,mcd_g], axis=1)
    print(data)
    #data = pd.concat([data[['RA','DEC','z','gmag','rmag','imag','zmag','Ymag']],data_dist,data_err], axis=1)

    #additional weight
    data['fuzzy_err'] = fuzzy_err(data_err)
    print(name," complite fuzzy_err")
    data_dist_r, max = fuzzy_dist(data_dist)
    data['fuzzy_dist'] = Normali(data_dist_r, max)
    print(name," complite fuzzy_dist")

    data.to_csv(f'{save_path}/{name}_main_sample.csv', index=False)
    return data

def data_concat(data1,data2,i,j):
    data1['Y'] = i
    data2['Y'] = j
    data12 = data1.append(data2, ignore_index=True)
    return data12

def data_begin(save_path,path_sample):
    def ff(name):
        
        data = pd.read_csv(f"{path_sample}/{name}_wol_full_phot_1021.csv", header=0, sep=',')
        #data = pd.read_csv(f"{path_sample}/{name}.csv", header=0, sep=',')
        #data = data.drop(['ExtClsCoad','ExtClsWavg'], axis=1)
        data = data.fillna(0)
        print(data)
        #data.to_csv(f'{save_path}/data_{name}.csv', index = False)
        #data_exgal = pd.read_csv(f"{save_path}/data_exgal.csv", header=0, sep=',')
        data = process(data,name,save_path)
        '''     
        #data = pd.read_csv(f"{save_path}/{name}_main_sample.csv", header=0, sep=',')
        data = pd.read_csv(f"{save_path}/{name}_main_sample.csv", header=0, sep=',')
        if(not name == 'qso'):
            count_qso = 372097 #371016
            data = data.sample(count_qso, random_state = 1)
        '''
        return data
    '''
    data_exgal = pd.read_csv(f"{path_sample}/exgal.csv", header=0, sep=',')
    data_star = pd.read_csv(f"{path_sample}/star.csv", header=0, sep=',')

    data_exgal = data_exgal.drop(['ExtClsCoad','ExtClsWavg'], axis=1)
    data_star = data_star.drop(['ExtClsCoad','ExtClsWavg'], axis=1)
    
    #Check variable zero value
    data_exgal = data_exgal.fillna(0)
    data_star = data_star.fillna(0)
    
    min=1e9
    if (data_exgal.shape[0] > data_star.shape[0]):
        min = data_star.shape[0]
    else:
        min = data_exgal.shape[0]

    data_exgal = data_exgal.sample(min, random_state=1)
    data_star = data_star.sample(min, random_state=1)

    data_exgal = data_exgal.reset_index(drop=True)
    data_star = data_star.reset_index(drop=True)

    print(data_exgal)
    data_exgal.to_csv(f'{save_path}/data_exgal.csv', index = False)
    data_star.to_csv(f'{save_path}/data_star.csv', index = False)
    
    #data_exgal = pd.read_csv(f"{save_path}/data_exgal.csv", header=0, sep=',')
    #data_star = pd.read_csv(f"{save_path}/data_star.csv", header=0, sep=',')

    data_exgal = process(data_exgal,"exgal",save_path)
    data_star = process(data_star,"star",save_path)
    '''
    
    data1 = ff('star')
    data2 = ff('qso')
    data3 = ff('gal')
    
    #data_exgal = pd.read_csv(f"{save_path}/exgal_main_sample.csv", header=0, sep=',')
    #data_star = pd.read_csv(f"{save_path}/star_main_sample.csv", header=0, sep=',')

    
    data1['star_cls'], data1['qso_cls'], data1['gal_cls'] = 1,0,0
    data2['star_cls'], data2['qso_cls'], data2['gal_cls'] = 0,1,0
    data3['star_cls'], data3['qso_cls'], data3['gal_cls'] = 0,0,1

    data12 = pd.concat([data1,data2], ignore_index=True)
    data123 = pd.concat([data12,data3], ignore_index=True)
    data123 = data123.sample(data123.shape[0], random_state=1)

    #delta_1_0 = data_concat(data_exgal,data_star,1,0)
    #delta_0_1 = data_concat(data_exgal,data_star,0,1)

    #return delta_1_0, delta_0_1
    data123.to_csv(f'{save_path}/all.csv',index = False)
    return data123

def data_phot_hist(path_sample = '/home/lrikozavr/ML_work/des_pro/ml/data', save_path = '/home/lrikozavr/ML_work/des_pro/ml/pictures/hist'):
    features = ['gmag&rmag', 'gmag&imag', 'gmag&zmag', 'gmag&Ymag', 
    'rmag&imag', 'rmag&zmag', 'rmag&Ymag', 
    'imag&zmag', 'imag&Ymag', 
    'zmag&Ymag',
    'gmag','rmag','imag','zmag','Ymag']
    class_name_mass = ['star','qso','gal']
    from grafics import Hist1
    for name in class_name_mass:
        data = pd.read_csv(f"{path_sample}/{name}_main_sample.csv", header=0, sep=',')
        for name_phot in features:
            Hist1(data[name_phot],save_path,f'{name}_{name_phot}',name_phot)
#data_phot_hist()