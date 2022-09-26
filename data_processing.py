#!/home/kiril/python_env_iron_ment/new_proj/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os

from fuzzy_options import fuzzy_dist,fuzzy_err, Normali, colors, MCD

def NtoPtoN(data,index):
    res = []
    for i in index:
        res.append(data.iloc[i])
    res = pd.DataFrame(np.array(res), columns=data.columns.values)
    return res

def process(data,name,save_path):
    #data_mags = data.drop(['RA','DEC','z','CatName','Class','Y'], axis=1)
    data_mags = data.drop(['RA','DEC','z','Y'], axis=1)
    data_dist, data_err = colors(data_mags)
    print(name," complite colors")
    dfsg = MCD(data_dist,0)
    print(name," complite MCD")

    data = NtoPtoN(data,dfsg)
    data_dist = NtoPtoN(data_dist,dfsg)
    data_err = NtoPtoN(data_err,dfsg)

    #data = pd.concat([data[['RA','DEC','z','CatName','Class','Y']],data_dist,data_err], axis=1)
    data = pd.concat([data[['RA','DEC','z','Y']],data_dist,data_err], axis=1)

    #data['fuzzy_err'] = fuzzy_err(data_err)
    #print(name," complite fuzzy_err")
    #data_dist_r, max = fuzzy_dist(data_dist)
    #data['fuzzy_dist'] = Normali(data_dist_r, max)
    #print(name," complite fuzzy_dist")

    data.to_csv(f'{save_path}/{name}_main_sample.csv', index=False)
    return data

#def data_concat(data1,data2,data3,i,j,k):
def data_concat(data1,data2,i,j):
    data1['Y'] = i
    data2['Y'] = j
    #data3['Y'] = k
    data12 = data1.append(data2, ignore_index=True)
    #data123 = data12.append(data3, ignore_index=True)
    #return data123
    return data12

def data_begin(save_path,path_sample):
    #path_sample = "/media/kiril/j_08/ML/extragal"

    #data_qso = pd.read_csv(f"{path_sample}/qso_main_sample.csv", header=0, sep=',')
    #data_gal = pd.read_csv(f"{path_sample}/gal_main_sample.csv", header=0, sep=',')
    #data_star = pd.read_csv(f"{path_sample}/star_main_sample.csv", header=0, sep=',')

    
    #path_sample = '/home/kiril/github/z'

#    data_qso = pd.read_csv(f"{path_sample}/qso_sample.csv", header=0, sep=',')
    #data_gal = pd.read_csv(f"{path_sample}/sample_gal.csv", header=0, sep=',')
#    data_star = pd.read_csv(f"{path_sample}/star_sample.csv", header=0, sep=',')

#    data_qso['Y'] = "1"
    #data_gal['Y'] = "1"
#    data_star['Y'] = "0"
    #Check variable zero value
    #data.fillna(0)
    
    #l = [len(data_gal), len(data_qso), len(data_star)]
    #l = [len(data_qso), len(data_star)]
    #min=1e+9
    #for i in l:
    #    if(i < min):
    #        min=i
    #print('MIN.....',min)   
#    min = 377679
    #data_gal = data_gal.sample(min, random_state=1)
    #data_qso = data_qso.sample(min, random_state=1)
    #data_star = data_star.sample(min, random_state=1)

    #data_gal = data_gal.reset_index(drop=True)
    #data_qso = data_qso.reset_index(drop=True)
    #data_star = data_star.reset_index(drop=True)

    #print(data_qso)
    #data_qso.to_csv(f'{save_path}/data_qso.csv', index = False)
    #data_star.to_csv(f'{save_path}/data_star.csv', index = False)
    
    data_qso = pd.read_csv(f"{save_path}/qso_main_sample.csv", header=0, sep=',')
    data_star = pd.read_csv(f"{save_path}/star_main_sample.csv", header=0, sep=',')



    #data_gal = process(data_gal,"gal",save_path)
    
    #data_qso = process(data_qso,"qso",save_path)
    #data_star = process(data_star,"star",save_path)
    

    #data_1_0_0 = data_concat(data_gal,data_qso,data_star,1,0,0)
    #data_0_1_0 = data_concat(data_gal,data_qso,data_star,0,1,0)
    #data_0_0_1 = data_concat(data_gal,data_qso,data_star,0,0,1)
    
    #return data_1_0_0, data_0_1_0, data_0_0_1

    delta_1_0 = data_concat(data_qso,data_star,1,0)
    delta_0_1 = data_concat(data_qso,data_star,0,1)
    
    return delta_1_0, delta_0_1
