#!/home/kiril/python_env_iron_ment/new_proj/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os

from fuzzy_options import fuzzy_dist,fuzzy_err, Normali, colors, MCD

path_sample = '/home/kiril/github/z'
'''
agn_name = ['agn_type_1','agn_type_2','blazar']
sfr_name = 'sfg'
qso_name = 'qso'
star_name = 'star'
'''
data_qso = pd.read_csv(f"{path_sample}/sample_qso.csv", header=0, sep=',')
data_gal = pd.read_csv(f"{path_sample}/sample_gal.csv", header=0, sep=',')
data_star = pd.read_csv(f"{path_sample}/sample_star.csv", header=0, sep=',')

data_qso['Y'] = "2"
data_gal['Y'] = "1"
data_star['Y'] = "0"

data.fillna(0)


def NtoPtoN(data,index):
    res = []
    for i in index:
        res.append(data.iloc[i])
    res = pd.DataFrame(np.array(res), column=data.columns.values)
    return res

def process(data):
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

    data.to_csv('main_sample.csv', index=False)

process(data_gal)
process(data_qso)
process(data_star)


training_data = data.sample(20000, random_state=1)
training_data.to_csv('training_sample_sfg.csv', index=False)