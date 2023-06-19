# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

#Середнє, дисперсія і т.д. для всіх значень метрик отриманих після Kfold
from data_processing import M, D

def metric_stat(data):
    res = pd.DataFrame()
    for index, name in enumerate(data.columns.values):
        average = M(data[name],data.shape[0])
        dispersion = D(data[name],data.shape[0])
        max = data[name].max()
        min = data[name].min()
        df = pd.DataFrame(np.array([average,dispersion,max,min]), columns=data.columns.values, index=["average","dispersion","max","min"])
        res = pd.concat([res,df], axis = 1)
        '''
        if (index < 1):
            res = df
        else:
            res = pd.concat([res,df], axis = 1)
        '''
    return res

