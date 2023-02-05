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
    #data_mags = data.drop(['RA','DEC','z','CatName','Class'], axis=1)
    data_mags = data.drop(['RA','DEC','z'], axis=1)
    data_dist, data_err = colors(data_mags)
    print(name," complite colors")
    dfsg = MCD(data_dist,0)
    print(name," complite MCD")

    data = NtoPtoN(data,dfsg)
    data_dist = NtoPtoN(data_dist,dfsg)
    data_err = NtoPtoN(data_err,dfsg)

    #data = pd.concat([data[['RA','DEC','z','CatName','Class']],data_dist,data_err], axis=1)
    data = pd.concat([data[['RA','DEC','z']],data_dist,data_err], axis=1)

    #additional weight
    #data['fuzzy_err'] = fuzzy_err(data_err)
    #print(name," complite fuzzy_err")
    #data_dist_r, max = fuzzy_dist(data_dist)
    #data['fuzzy_dist'] = Normali(data_dist_r, max)
    #print(name," complite fuzzy_dist")

    data.to_csv(f'{save_path}/{name}_main_sample.csv', index=False)
    return data

def data_concat(data1,data2,i,j):
    data1['Y'] = i
    data2['Y'] = j
    data12 = data1.append(data2, ignore_index=True)
    return data12

def data_begin(save_path,path_sample):
    data_exgal = pd.read_csv(f"{path_sample}/exgal.csv", header=0, sep=',')
    data_star = pd.read_csv(f"{path_sample}/star.csv", header=0, sep=',')

    #Check variable zero value
    data_exgal = data_exgal.fillna(0)
    data_star = data_star.fillna(0)
    
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

    #data_exgal = pd.read_csv(f"{save_path}/exgal_main_sample.csv", header=0, sep=',')
    #data_star = pd.read_csv(f"{save_path}/star_main_sample.csv", header=0, sep=',')

    delta_1_0 = data_concat(data_exgal,data_star,1,0)
    delta_0_1 = data_concat(data_exgal,data_star,0,1)
    
    return delta_1_0, delta_0_1
