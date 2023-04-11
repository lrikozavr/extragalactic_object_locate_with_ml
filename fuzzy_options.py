# -*- coding: utf-8 -*-

from numpy.core.defchararray import count
import pandas as pd
import numpy as np
import math

delta = 1e-7

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

def Gauss_cut(data,n):
    m=M(data,n)
    d=D(data,n)
    #
    rezult = []
    
    for i in range(n):
        if(math.e**(-((data[i]-m)**2)/(2*d**2))/(d*math.sqrt(2*math.pi)) > 0.005):
            rezult.append(i)
            
    return rezult

def Normali(data,max):
    count = len(data)
    s = []
    for i in range(count):
        s.append(1 - data[i] / (max + delta))
    return s

def fuzzy_dist(data):
    columns = data.columns.values
    count = len(data)
    rc = pd.DataFrame(np.array(['fuf']), columns=['word'])
    #
    for col in columns:
        rc[col] = M(data[col],count)
        #print(rc[col])
        #print("M        ", M(data[col],count))
    #print("rc           ",rc)
    r = []
    max = -1
    for i in range(count):
        ev_sum = 0
        for col in columns:
            ev_sum += (rc[col].iloc[0] - data[col].iloc[i])**2
        #print(ev_sum)    
        r.append(math.sqrt(ev_sum))
        if(r[i] > max):
            max = r[i]
    #print("fuzzy_dist complite")
    return r, max

#?????????????????????????????????????????????????????????
def fuzzy_err(data):
    columns = data.columns.values
    count = len(data)
    #print(count,columns)

    summ = []
    max = np.zeros(len(columns))
    index=0
    for col in columns:
        for i in range(count):
            if(data[col].iloc[i] > max[index]):
                max[index]=data[col].iloc[i]
        index+=1
    np.zeros((count,))

    for i in range(count):
        sum = 0
        index = 0
        for col in columns:
            sum += (1 - data[col].iloc[i]/(max[index]+delta))**2
            index += 1
        summ.append(math.sqrt(sum/float(index)))
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
    print(colours_name)
    print(colours_error_name)
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
    


def MCD(data,deep_i):
    print(deep_i)
    deep_i+=1
    n = data.shape[0]
    count_col = len(data.columns.values)
    t0 = T0(data,n)
    #print(t0)
    s0 = S0(data,t0,n)
    
    #print("s0000000000",s0)
    if (np.linalg.det(s0) == 0):
        print("det S0 = 0. MCD Complite")
        print(deep_i)
        return data
    d = np.zeros(n)

    max, index = 0, 0
    for i in range(n):
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
        #if (d[i] > max):
        #    max = d[i]
        #    index = i
    '''
    ar_i = [i for i in range(n)]
    
    from matplotlib import pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    fig.set_size_inches(10,10)
    ax.scatter(ar_i,d)
    fig.savefig(f"{deep_i}_d.png")
    plt.close(fig)
    '''
    index_d = Gauss_cut(d,n)

    '''
    rezult = []
    for i in index_d:
        rezult.append(data.iloc[i])

    rezult = pd.DataFrame(np.array(rezult),columns=data.columns.values)
    
    return rezult
    '''
    return index_d
    '''
    t1=T0(data,n-1)

    s1=S0(data,t1,n-1)

    if (np.linalg.det(s0) <= np.linalg.det(s1)):
        print("det S0 = det S1. MCD Complite")
        print(deep_i)
        return data
    else:
        MCD(data,deep_i)
    '''
    #MCD(data,deep_i)

def Ell(data):
    n = data.shape[0]
    count_col = len(data.columns.values)
    center = T0(data,n)

    max = 0
    summ = np.zeros(n)
    for i in range(n):
        sum=0
        jj=0
        for j in data.columns.values:
            sum += (data[j].iloc[i] - center[jj])**2
            jj+=1
        summ[i]=math.sqrt(sum)
        if (sum>max):
            max = sum
    index = np.argsort(summ)
    #summ = np.sort(summ)
    #?
    print('max was found')
    f_i = np.zeros(count_col)
    for i in range(count_col):
        f_i[i] = index[n-i-1]
    
    R = max
    mat = [[0]*count_col for i in range(count_col)]
    ii=0
    for i in f_i:
        jj=0
        for j in data.columns.values:
            mat[ii][jj] = (data[j].iloc[int(i)] - center[jj])**2
            jj+=1
        ii+=1
    mat_inv = np.linalg.inv(mat)
    print('inv matrix was found')
    coef = np.zeros(count_col)

    for i in range(count_col):
        sum = 0
        for j in range(count_col):
            sum += mat_inv[i][j]*R
        coef[i] = sum
    
    return center, coef
    #(x[i]-center[i])^2/coef[i]+...=R^2
    #for(i=1;i<15;i+=1){}

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
    
