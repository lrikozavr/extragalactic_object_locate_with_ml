# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

from numpy.core.defchararray import count
import pandas as pd
import numpy as np
import math
import os

DELTA = 1e-7

def M(data,n):
    sum = 0
    #data = data.reset_index(drop=True)
    #print(data)
    for i in data:
        sum += i
    return sum/float(n)

def D(data,n):
    m=M(data,n)
    sum=0
    for i in data:
        sum += (i - m)**2
    return math.sqrt(sum/float(n))

def Gauss_cut(data,n,threshold = 0.005):
    m=M(data,n)
    d=D(data,n)
    #
    gauss = np.zeros(n)
    for i in range(n):
        gauss[i] = math.e**(-((data[i]-m)**2)/(2*d**2))/(d*math.sqrt(2*math.pi))
    #
    outlire = []
    
    for i in range(n):
        if(gauss[i] < threshold):
            outlire.append(i)

    return gauss, outlire

def Normali(data,max):
    count = len(data)
    s = np.zeros(count)
    for i in range(count):
        s[i] = 1 - data[i] / (max + DELTA)
    return s

def fuzzy_dist(data):
    columns = data.columns.values
    count = data.shape[0]
    rc = pd.DataFrame()
    #
    for col in columns:
        rc[col] = M(data[col],count)
        #print(rc[col])
        #print("M        ", M(data[col],count))
    #print("rc           ",rc)
    r = np.zeros(data.shape[0])
    max = -1
    for i in range(count):
        ev_sum = 0
        for col in columns:
            ev_sum += (rc[col].iloc[0] - data[col].iloc[i])**2
        #print(ev_sum)    
        r[i] = math.sqrt(ev_sum)
        if(r[i] > max):
            max = r[i]
    #print("fuzzy_dist complite")
    return r, max

#?????????????????????????????????????????????????????????
def fuzzy_err(data):
    columns = data.columns.values
    count = data.shape[0]
    #print(count,columns)

    max = np.zeros(len(columns))
    index=0
    for col in columns:
        max[index] = data[col].max()
        index+=1

    summ = np.zeros(count)
    for i in range(count):
        sum = 0
        index = 0
        for col in columns:
            sum += (1 - data[col].iloc[i]/(max[index]+DELTA))**2
            index += 1
        summ[i] = math.sqrt(sum/float(index))
    #print("fuzzy_err complite")
    return summ

def colors(data):
    print(data)
    list_name = data.columns.values
    count = data.shape[0]
    mags = int(data.shape[1]/2)
    num_colours = sum(i for i in range(mags))
    colours = np.zeros((count,num_colours))
    colours_error = np.zeros((count,num_colours))
    index = 0
    colours_name, colours_error_name = [], []
    data=np.array(data)
    for j in range(mags):
        for i in range(j, mags):
            if(i!=j):
                colours_name.append(f"{list_name[j*2]}&{list_name[i*2]}")
                colours_error_name.append(f"{list_name[j*2+1]}&{list_name[i*2+1]}")
                colours[:,index] = data[:,j*2] - data[:,i*2]
                colours_error[:,index] = np.sqrt(data[:,j*2+1]**2 + data[:,i*2+1]**2)
                index += 1
    #print(colours_name)
    #print(colours_error_name)
    colours = pd.DataFrame(colours, columns=colours_name)
    colours_error = pd.DataFrame(colours_error, columns=colours_error_name)
    return colours, colours_error

def T0(data,n):
    vec = []
    for j in data.columns.values:
        vec.append(M(data[j],n))
    return np.array(vec)            

def S0(data,t0,n):
    count_col = len(data.columns.values)
    mat = np.zeros((count_col,count_col))

    for i in range(n):
        x = np.array(data.iloc[i])
        matrix = x - t0       
        for j1 in range(count_col):
            for j2 in range(count_col):
                mat[j1][j2] += matrix[j1]*matrix[j2]
        
    #print("MAT",mat)
    for j1 in range(count_col):
        for j2 in range(count_col):
            mat[j1][j2] = mat[j1][j2] / float(n)
    #print("MAT/////////////",mat)
    #print(n)
    return np.linalg.inv(mat)
    


def MCD(data,deep_i,config):
    print(deep_i)
    deep_i+=1
    count = data.shape[0]
    count_col = data.shape[1]
    t0 = T0(data,count)
    #print(t0)
    s0 = S0(data,t0,count)
    
    if (np.linalg.det(s0) == 0):
        print("det S0 = 0. MCD Complite")
        #print(deep_i)
        return data
    
    d = np.zeros(count)

    for i in range(count):
        x = np.array(data.iloc[i])
        matrix = x - t0
        res=0
        for j1 in range(count_col):
            sum = 0
            for j2 in range(count_col):
                sum += matrix[j2]*s0[j2][j1]
            res += sum*matrix[j1]
        if(res < 0):
            continue
        d[i] = math.sqrt(res)

    gauss, outlire = Gauss_cut(d,count,threshold=config.flags['data_preprocessing']['main_sample']['outlire']['add_param']['additional_parametr'])

    return d, gauss, outlire
'''
def redded_des(data):
    def dust_SFD(ra,dec):
        from dustmaps.config import config
        config['data_dir'] = '/home/lrikozavr/catalogs/dustmaps/'
        from astropy.coordinates import SkyCoord
        from astropy import units as u
        from dustmaps.sfd import SFDQuery
        coords = SkyCoord(ra=ra, dec=dec, unit=(u.degree,u.degree))
        coords.galactic
        sfd = SFDQuery()
        rezult = sfd(coords)
        return rezult
    data['E(B-V)'] = dust_SFD(data['RA'],data['DEC'])
    data['gmag'] -= 3.186*data['E(B-V)']
    data['rmag'] -= 2.140*data['E(B-V)']
    data['imag'] -= 1.5689*data['E(B-V)']
    data['zmag'] -= 1.1959*data['E(B-V)']
    data['Ymag'] -= 1.048*data['E(B-V)']
'''
def redded_(data,name):
    return


def NtoPtoN(data,index):
    res = []
    for i in index:
        res.append(data.iloc[i])
    res = pd.DataFrame(np.array(res), columns=data.columns.values)
    return res

def process(path_sample,name,save_path, config):
    #data_mags = data.drop(['RA','DEC','z','CatName','Class'], axis=1)
    data = pd.read_csv(f"{path_sample}/{name}.csv", header=0, sep=',')
    print(f"read data {name}")

    #Check variable zero value
    data = data.fillna(0)
    base = ['RA','DEC','z']
    def data_issue(check):
        match check:
            case 'err':
                if(config.flags['data_preprocessin']['main_sample']['color']['work']):
                    return data_err
                else:
                    raise Exception('cant made outlire by err, \ncheck flags["data_preprocessin"]["main_sample"]["color"]["work"] in config')
            case 'color':
                if(config.flags['data_preprocessin']['main_sample']['color']['work']):
                    return data_color
                else:
                    raise Exception('cant made outlire by color, \ncheck flags["data_preprocessin"]["main_sample"]["color"]["work"] in config')
            case 'features':
                return data[config.features]
            case _:
                raise Exception('wrong value flags["data_preprocessin"]["main_sample"]["weight"]["value"]')

    #redded_des(data)
    #print(name, 'deredded complite')
    if(config.flags['data_preprocessin']['main_sample']['color']['work']):
        data_color, data_err = colors(data[config.features])
        print(name," complite colors")
        if(config.flags['data_preprocessin']['main_sample']['color']['mags']):
            data = pd.concat([data,data_color],axis=1)
        if(config.flags['data_preprocessin']['main_sample']['color']['err']):
            data = pd.concat([data,data_err],axis=1)
        
    if(config.flags['data_preprocessin']['main_sample']['outlire']['cut']):
        if( "MCD" in config.flags['data_preprocessin']['main_sample']['outlire']['method'] ):
            mcd_d, gauss_d, outlire = MCD(data_issue(config.flags['data_preprocessin']['main_sample']['outlire']['value']),0,config)
            print(name," complite MCD")
            if(config.flags['data_preprocessin']['main_sample']['outlire']['add_param']['add']):
                mcd_d = pd.DataFrame(np.array(mcd_d), columns = ['mcd_d'])
                mcd_g = pd.DataFrame(np.array(gauss_d), columns = ['mcd_g'])
                data = pd.concat([data,mcd_d,mcd_g], axis=1)
            if(config.flags['data_preprocessin']['main_sample']['outlire']['cut']):
                data = data.drop(outlire)
    
    
    #additional weight
    if('fuzzy_err' in config.flags['data_preprocessin']['main_sample']['weight']['method']):        
        index = config.flags['data_preprocessin']['main_sample']['weight']['method'].index('fuzzy_err')
        data['fuzzy_err'] = fuzzy_err(data_issue(config.flags['data_preprocessin']['main_sample']['weight']['value'][index]))
        print(name," complite fuzzy_err")

    if('fuzzy_dist' in config.flags['data_preprocessin']['main_sample']['weight']['method']):    
        index = config.flags['data_preprocessin']['main_sample']['weight']['method'].index('fuzzy_dist')
        data_dist, max = fuzzy_dist(data_issue(config.flags['data_preprocessin']['main_sample']['weight']['value'][index]))
        data['fuzzy_dist'] = Normali(data_dist, max)
        print(name," complite fuzzy_dist")

    data.to_csv(f'{save_path}/{name}_main_sample.csv', index=False)
    return data

def data_preparation(save_path,path_sample,name_class,config):

    def preparation(name):
        data = pd.DataFrame()
        if(not config.flags['data_preprocessing']['main_sample']['work']):
            if(os.path.isfile(f"{save_path}/{name}_main_sample.csv")):   
                data = pd.read_csv(f"{save_path}/{name}_main_sample.csv", header=0, sep=',')
            else:
                data = process(path_sample,name,save_path,config)
                data.to_csv(f'{save_path}/{name}_main_sample.csv', index=False)
        else:
            data = process(path_sample,name,save_path,config)
            data.to_csv(f'{save_path}/{name}_main_sample.csv', index=False)

        return data
    
    count = np.zeros(len(name_class))
    data_mass = []
    
    data = pd.DataFrame()
    
    for n, name in enumerate(name_class):
        data_temp = preparation(name)
        for n_, name_ in enumerate(name_class):
            if(n_ == n):
                data_temp[f'{name_}_cls'] = 1
            else:
                data_temp[f'{name_}_cls'] = 0
        count[n] = data_temp.shape[0]
        if(config.flags['data_preprocessing']['balanced']):
            data_mass.append(data_temp)
        else:
            data = pd.concat([data,data_temp], ignore_index=True)

    if(config.flags['data_preprocessing']['balanced']):
        for i in range(name_class):
            data_temp = data_mass[i].sample(count.min(), random_state = 1)
            data = pd.concat([data,data_temp], ignore_index=True)
        
    del data_mass
    
    for i in range(len(name_class)):
        print(f"{name_class[i]} count:\t---\t", count[i])

    data.to_csv(f'{save_path}/all.csv',index = False)

    '''
    data1 = preparation('star')
    data2 = preparation('qso')
    data3 = preparation('gal')
    
    #data_exgal = pd.read_csv(f"{save_path}/exgal_main_sample.csv", header=0, sep=',')
    #data_star = pd.read_csv(f"{save_path}/star_main_sample.csv", header=0, sep=',')

    data1['star_cls'], data1['qso_cls'], data1['gal_cls'] = 1,0,0
    data2['star_cls'], data2['qso_cls'], data2['gal_cls'] = 0,1,0
    data3['star_cls'], data3['qso_cls'], data3['gal_cls'] = 0,0,1

    data12 = pd.concat([data1,data2], ignore_index=True)
    data123 = pd.concat([data12,data3], ignore_index=True)
    data123 = data123.sample(data123.shape[0], random_state=1)
    
    data123.to_csv(f'{save_path}/all.csv',index = False)
    '''
    
    return data
