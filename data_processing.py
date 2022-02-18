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
    res = pd.DataFrame(np.array(res), column=data.columns.values)
    return res

def process(data,name,save_path):
    data_mags = data.drop(['RA','DEC','z','CatName','Class','Y'], axis=1)
    data_dist, data_err = colors(data_mags)
    dfsg = MCD(data_dist,0)
    
    data = NtoPtoN(data,dfsg)
    data_dist = NtoPtoN(data_dist,dfsg)
    data_err = NtoPtoN(data_err,dfsg)

    data = pd.concat([data[['RA','DEC','z','type','name','Y']],data_dist,data_err], axis=1)

    data['fuzzy_err'] = fuzzy_err(data_err)
    data_dist_r, max = fuzzy_dist(data_dist)
    data['fuzzy_dist'] = Normali(data_dist_r, max)

    data.to_csv(f'{save_path}/{name}_main_sample.csv', index=False)
    return data

def data_begin(save_path):
    path_sample = '/home/kiril/github/z'

    data_qso = pd.read_csv(f"{path_sample}/sample_qso.csv", header=0, sep=',')
    data_gal = pd.read_csv(f"{path_sample}/sample_gal.csv", header=0, sep=',')
    data_star = pd.read_csv(f"{path_sample}/sample_star.csv", header=0, sep=',')

    data_qso['Y'] = "2"
    data_gal['Y'] = "1"
    data_star['Y'] = "0"
    #Check variable zero value
    #data.fillna(0)

    data_gal = process(data_gal,"gal",save_path)
    data_qso = process(data_qso,"qso",save_path)
    data_star = process(data_star,"star",save_path)

    l = [len(data_gal), len(data_qso), len(data_star)]
    min=1e+9
    for i in l:
        if(i < min):
            min=i

    data_gal = data_gal.sample(min, random_state=1)
    data_qso = data_qso.sample(min, random_state=1)
    data_star = data_star.sample(min, random_state=1)

    data_gal_qso = data_gal.append(data_qso, ignore_index=True)
    data_gal_qso_star = data_gal_qso.append(data_star, ignore_index=True)
    return data_gal_qso_star
