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
    #print(data)
    #
    for col in columns:
        rc[col] = [M(data[col],count)]
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
    #index=0
    for index, col in enumerate(columns):
        max[index] = data[col].max()
        

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

def flux_var(data):
    count = data.shape[0]
    mags = int(data.shape[1]//3)
    list_name = data.columns.values

    data = np.array(data)

    #print(data)
    data_var = np.empty((count,mags))
    data_var_name = []
    for j in range(0,mags*3,3):
        data_var[:,j//3] = np.power(data[:,j],0.5)*(data[:,j+2]/data[:,j+1])
        data_var_name.append(f"{list_name[j+1]}_var")
    data_var = pd.DataFrame(data_var, columns=data_var_name)

    num_colours = sum(i for i in range(mags))
    data_color = np.empty((count,num_colours))
    data_color_name = []
    index=0
    for j in range(mags):
        for i in range(j, mags):
            if(i!=j):
                data_color_name.append(f"{list_name[j*3+1]}&{list_name[i*3+1]}")
                data_color[:,index] = data[:,j*3+1]/(data[:,j*3+2]*data[:,j*3]) - data[:,i*3+1]/(data[:,i*3+2]*data[:,i*3])
                index += 1                
    #print(data_color_name,data_var_name)
    data_color = pd.DataFrame(data_color, columns=data_color_name)
    #print(data_var, data_color)
    return data_var, data_color

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
    #print(deep_i)
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

def NtoPtoN(data,index):
    res = []
    for i in index:
        res.append(data.iloc[i])
    res = pd.DataFrame(np.array(res), columns=data.columns.values)
    return res

def get_features(features_list,config):
    features = []

    colours_name, colours_error_name = [], []
    mags_name, mags_error_name = [], []
    flux_color = []
    flux_var_name = []

    photometry_list = config.features["data"]["photometry"]
    flux_list = config.features["data"]["flux"]

    mags_count = int(len(photometry_list)/2)
    for j in range(mags_count):
        for i in range(j, mags_count):
            if(i!=j):
                colours_name.append(f"{photometry_list[j*2]}&{photometry_list[i*2]}")
                colours_error_name.append(f"{photometry_list[j*2+1]}&{photometry_list[i*2+1]}")
                flux_var_name.append(f"{flux_list[j*3+1]}_var")
                flux_color.append(f"{flux_list[j*3+1]}&{flux_list[i*3+1]}")
        mags_name.append(f"{photometry_list[j*2]}")
        mags_error_name.append(f"{photometry_list[j*2+1]}")

    def check_features(features_list):
        temp_features_list = []
        try:
            for name in features_list.keys():
                temp_features_list.extend(features_list[name])
                #print(features_list[name][0])
            return temp_features_list
        except:
            return features_list

    local_features_list = check_features(features_list)
    #print(local_features_list)

    for features_flag in local_features_list:
        match features_flag:
            #може потрібно тут ставити запобіжник?
            case "color":
                features.extend(colours_name)
                if not (config.flags['data_preprocessing']['main_sample']['color']['work'] and config.flags['data_preprocessing']['main_sample']['color']['mags']):
                    print("WARNING: data may don't have a property columns, check: \nconfig.flags['data_preprocessing']['main_sample']['color']['work']\nand\nconfig.flags['data_preprocessing']['main_sample']['color']['mags']\nand\nconfig.features['train']\n")
            case "mags":
                features.extend(mags_name)
            case "err_color":
                features.extend(colours_error_name)
                if not (config.flags['data_preprocessing']['main_sample']['color']['work'] and config.flags['data_preprocessing']['main_sample']['color']['err']):
                    print("WARNING: data may don't have a property columns\ncheck config.flags['data_preprocessing']['main_sample']['color']['work']\nand\nconfig.flags['data_preprocessing']['main_sample']['color']['err']\nand\nconfig.features['train']\n")
            case "err_mags":
                features.extend(mags_error_name)
            case "var":
                features.extend(flux_var_name)

            case "flux_color":
                features.extend(flux_color)

            case "raw":
                features.extend(config.features["data"]["astrometry"])
            case _:
                raise Exception('unknown config value config.features["train"]')
    
    return features

def deredded(data,config_local):
    mags = get_features(['mags'],config_local)

    from extinction_coefficient import extinction_coefficient
    EXTINCTION_COEFFICIENT_config_mass_value = pd.DataFrame()
    for i, name in enumerate(mags):
        EXTINCTION_COEFFICIENT_config_mass_value[name] = extinction_coefficient(config_local.flags["data_preprocessing"]["main_sample"]["deredded"]["coef"][i], mode='simple')

    if(not len(mags)==len(config_local.flags["data_preprocessing"]["main_sample"]["deredded"]["coef"])):
        raise Exception('config.flags["data_preprocessing"]["main_sample"]["deredded"]["coef"] invalid count')

    ra = data[config_local.base[0]].astype(float).values
    dec = data[config_local.base[1]].astype(float).values
    
    from dustmaps.config import config
    config['data_dir'] = config_local.flags["data_preprocessing"]["main_sample"]["deredded"]["dust_map_dir"]
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    from dustmaps.sfd import SFDQuery
    coords = SkyCoord(ra=ra, dec=dec, unit=(u.degree,u.degree))
    del ra, dec
    coords.galactic
    sfd = SFDQuery()
    rezult = sfd(coords)
    del coords
    
    data['E(B-V)'] = rezult
    del rezult

    threshold = config_local.flags["data_preprocessing"]["main_sample"]["deredded"]["threshold"]

    if(config_local.flags["data_preprocessing"]["main_sample"]["deredded"]["cut"]):
        data = data[data['E(B-V)'] < threshold] 
    else:
        data['E(B-V)'] = data['E(B-V)'].apply(lambda x: threshold if x > threshold else x)
    
    if(config_local.flags["data_preprocessing"]["main_sample"]["deredded"]["mode"] == "simple"):
        for name in mags:
            data[name] = data[name].astype(float) - data['E(B-V)']*EXTINCTION_COEFFICIENT_config_mass_value.loc[0,(name)]
    else:
        
        def culc(DATA,EBV,BP_RP,BAND):
            #print(EBV,BAND,BP_RP)
            return DATA - EBV*extinction_coefficient(Band=[BAND], EBV=[EBV], BP_RP=[BP_RP])
            

        culc_v = np.vectorize(culc)

        bp_rp = data[mags[len(mags)-2]].astype(float)-data[mags[len(mags)-1]].astype(float)
        
        MAX_WORKERS=len(mags)
        from concurrent.futures import ThreadPoolExecutor
        rezult = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for n, name in enumerate(mags):
                rezult.append(executor.submit(culc_v,data[name].astype(float),data['E(B-V)'].values,bp_rp,config_local.flags["data_preprocessing"]["main_sample"]["deredded"]["coef"][n]))

        for i, name in enumerate(mags):
            data[name] = rezult[i].result()       

        del bp_rp
    #data = data.drop(['E(B-V)'], axis=1)
    return data


def process(path_sample,name,save_path, config):
    #data_mags = data.drop(['RA','DEC','z','CatName','Class'], axis=1)
    data = pd.read_csv(f"{path_sample}/{name}.csv", header=0, sep=',')
    print(f"read data {name}")
    print(data)
    #Check variable zero value
    data = data.fillna(0)
            
    def data_issue(check):
        match check:
            case 'err':
                if(config.flags['data_preprocessing']['main_sample']['color']['work']):
                    return data_err
                else:
                    raise Exception('cant made outlire by err, \ncheck flags["data_preprocessing"]["main_sample"]["color"]["work"] in config')
            case 'color':
                if(config.flags['data_preprocessing']['main_sample']['color']['work']):
                    return data_color
                else:
                    raise Exception('cant made outlire by color, \ncheck flags["data_preprocessing"]["main_sample"]["color"]["work"] in config')
            case 'features':
                return data[config.features["data"]["photometry"]]
            case _:
                raise Exception('wrong value flags["data_preprocessing"]["main_sample"]["weight"]["value"]')
    
    #deredded
    if(config.flags['data_preprocessing']['main_sample']['deredded']['work']):
        data = deredded(data,config)
        print(name, "deredded complite")

    #range cut
    if(len(config.features["range"]["photometry"]) == len(config.features["data"]["photometry"]) // 2):
        for i in range(len(config.features["data"]["photometry"]) // 2):
           data = data[(data[config.features["data"]["photometry"][i*2]] > config.features["range"]["photometry"][i][0]) & (data[config.features["data"]["photometry"][i*2]] < config.features["range"]["photometry"][i][1])]
        data = data.reset_index()

    if(config.features["mod"] == "all"):
        if(config.flags['data_preprocessing']['main_sample']['flux']['work']):
            data_flux_var, data_flux_color = flux_var(data[config.features["data"]["flux"]])
            print(name," complite flux")
            if(config.flags['data_preprocessing']['main_sample']['flux']['var']):
                data = pd.concat([data,data_flux_var],axis=1)
            if(config.flags['data_preprocessing']['main_sample']['flux']['color']):
                data = pd.concat([data,data_flux_color],axis=1)
        else:
            data = data.drop(config.features["data"]["astrometry"].extend(config.features["data"]["flux"]), axis=1)



    #print(name, 'deredded complite')
    if(config.flags['data_preprocessing']['main_sample']['color']['work']):
        data_color, data_err = colors(data[config.features["data"]["photometry"]])
        print(name," complite colors")
        if(config.flags['data_preprocessing']['main_sample']['color']['mags']):
            data = pd.concat([data,data_color],axis=1)
        if(config.flags['data_preprocessing']['main_sample']['color']['err']):
            data = pd.concat([data,data_err],axis=1)


    if(config.flags['data_preprocessing']['main_sample']['outlire']['work']):
        if("MCD" in config.flags['data_preprocessing']['main_sample']['outlire']['method'] ):
            mcd_d, gauss_d, outlire = MCD(data_issue(config.flags['data_preprocessing']['main_sample']['outlire']['value']),0,config)
            print(name," complite MCD")
            if(config.flags['data_preprocessing']['main_sample']['outlire']['add_param']['add']):
                mcd_d = pd.DataFrame(np.array(mcd_d), columns = ['mcd_d'])
                mcd_g = pd.DataFrame(np.array(gauss_d), columns = ['mcd_g'])
                data = pd.concat([data,mcd_d,mcd_g], axis=1)
            if(config.flags['data_preprocessing']['main_sample']['outlire']['cut']):
                data = data.drop(outlire)
                data_color = data_color.drop(outlire)
                data_err = data_err.drop(outlire)
    
    
    #additional weight
    if('fuzzy_err' in config.flags['data_preprocessing']['main_sample']['weight']['method']):        
        index = config.flags['data_preprocessing']['main_sample']['weight']['method'].index('fuzzy_err')
        data['fuzzy_err'] = fuzzy_err(data_issue(config.flags['data_preprocessing']['main_sample']['weight']['value'][index]))
        print(name," complite fuzzy_err")

    if('fuzzy_dist' in config.flags['data_preprocessing']['main_sample']['weight']['method']):    
        index = config.flags['data_preprocessing']['main_sample']['weight']['method'].index('fuzzy_dist')
        data_dist, max = fuzzy_dist(data_issue(config.flags['data_preprocessing']['main_sample']['weight']['value'][index]))
        data['fuzzy_dist'] = Normali(data_dist, max)
        print(name," complite fuzzy_dist")


    data.to_csv(f'{save_path}/{config.name_main_sample}_{name}_main_sample.csv', index=False)

    return data

def data_preparation(save_path,path_sample,name_class,config):

    def preparation(name):
        data = pd.DataFrame()
        if(not config.flags['data_preprocessing']['main_sample']['work']):
            if(os.path.isfile(f"{save_path}/{config.name_main_sample}_{name}_main_sample.csv")):   
                data = pd.read_csv(f"{save_path}/{config.name_main_sample}_{name}_main_sample.csv", header=0, sep=',')
            else:
                data = process(path_sample,name,save_path,config)
        else:
            data = process(path_sample,name,save_path,config)

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
        for i in range(len(name_class)):
            data_temp = data_mass[i].sample(int(count.min()), random_state = 1)
            data = pd.concat([data,data_temp], ignore_index=True)
        print("data have equal count of classes")
    
    if(config.flags['data_preprocessing']['main_sample']['normalize']['work']):
        from sklearn.preprocessing import normalize
        #print(get_features(config.flags['data_preprocessing']['main_sample']['normalize']['features'],config))
        #print(data)
        features_normalize = get_features(config.flags['data_preprocessing']['main_sample']['normalize']['features'],config)
        data_values = data[features_normalize]
        columns = data_values.columns.values
        data_normalize, norms = normalize(data_values,norm=config.flags['data_preprocessing']['main_sample']['normalize']['mode'],axis=0,return_norm=True)
        data[features_normalize] = np.array(data_normalize)
        
        #### ????????????????????????????????????
        #data[features]-data[features].mean()
        #data[features]-data[features].std()
        ####
        
        #print(norms)
        #print(np.array(norms).transpose())
        pd.DataFrame(np.array([norms]),columns = columns).to_csv(f"{config.path_stat}/{config.name_main_sample}_norms.csv", index=False)
        print("complite normalize")
    
    del data_mass
    
    for i in range(len(name_class)):
        print(f"{name_class[i]} count:\t---\t", int(count[i]))

    print("final sample:\t---\t", data.shape[0])

    data = data.sample(data.shape[0], ignore_index=True)
    data.to_csv(f'{save_path}/{config.name_main_sample}_all.csv',index = False)
    

    return data
