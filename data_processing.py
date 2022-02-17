#!/home/kiril/python_env_iron_ment/new_proj/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os

from fuzzy_options import fuzzy_dist,fuzzy_err, Normali, colors

path_sample = '/home/kiril/github/z/sample_extragal_phot.csv'
agn_name = ['agn_type_1','agn_type_2','blazar']
sfr_name = 'sfg'
qso_name = 'qso'
star_name = 'star'

data = pd.read_csv(f".csv", header=0, sep=',')

data = data.drop(['Name'], axis=1)
#print(data)

#Отсекаем изначально ненужное (Значения часто пустые)
#data = data.drop(['e_Jmag','e_Hmag','e_Kmag','Jmag','Hmag','Kmag','e_W4mag','W4mag',
#                    'parallax','pmra','pmdec','parallax_error','pm','pmra_error','pmdec_error','bp_rp'], axis=1)
#data.fillna(0)
#для fuzzy_err

data_mags = data.drop(['RA','DEC','z','type','name','Y'], axis=1)

data_dist, data_err = colors(data_mags)
from fuzzy_err_calc import MCD
dfsg = MCD(data_dist,0)
dfsg.to_csv("MCD.csv")
exit()

data = pd.concat([data[['RA','DEC','z','type','name','Y']],data_dist,data_err], axis=1)
data_dist['Y'] = data['Y']
#data_err = data.drop(['RA','DEC','z','type','name','W1mag','W2mag','W3mag','Y',
#                    #'gmag','rmag','imag','zmag','ymag',
#                    'phot_g_mean_mag','phot_bp_mean_mag','phot_rp_mean_mag'], axis=1)
#для fuzzy_dist
#data_dist = data.drop(['RA','DEC','z','type','name','e_W1mag','e_W2mag','e_W3mag',
#                    #'e_gmag','e_rmag','e_imag','e_zmag','e_ymag',
#                    'phot_g_mean_mag_error','phot_bp_mean_mag_error','phot_rp_mean_mag_error'], axis=1)

#print(data_err)

data['fuzzy_err'] = fuzzy_err(data_err)

data_dist_1 = data_dist[data_dist['Y'] == 1]
data_dist_0 = data_dist[data_dist['Y'] == 0]

data_dist_1 = data_dist_1.drop(['Y'], axis=1)
data_dist_0 = data_dist_0.drop(['Y'], axis=1)

data_dist_1, max = fuzzy_dist(data_dist_1)
dat1 = pd.DataFrame(np.array(Normali(data_dist_1, max)))

data_dist_0, max = fuzzy_dist(data_dist_0)
dat0 = pd.DataFrame(np.array(Normali(data_dist_0, max)))

data['fuzzy_dist'] = dat1.append(dat0, ignore_index=True)

data.to_csv('main_sample_sfg.csv', index=False)

training_data = data.sample(20000, random_state=1)
training_data.to_csv('training_sample_sfg.csv', index=False)