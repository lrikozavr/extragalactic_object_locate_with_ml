# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os


#sample_path = f'{general_path}/sample'

def diff_class(data_path,name_class,path_sample,base):
    #create file for class obj
    #data_mass = []
    #for i in range(len(name_class)):
    #    data_mass.append(pd.DataFrame())
    #
    data_file = []
    for i in range(len(name_class)):
        f = open(f'{path_sample}/{name_class[i]}_origin.csv','w')
        data_file.append(f)

    #extract exgal and star obj from file exit.sort
    count = np.zeros(len(name_class))

    base_index_array = [0]
    for line in open(data_path):
        n = line.split(',')
        if(base_index_array[0] == 0):
            base_index_array = [n.index(base[i]) for i in range(len(base))]
        base_array = [n[base_index_array[i]] for i in range(len(base))]
        #if(len(n[4].split('_')) == 1 and (n[4] == 'lamost' or n[4] == 'sdss')):
        index = 0 
        line_out = ''
        for word in base_array:
            if(not index == len(base_index_array) - 1):
                line_out += word + ','
            else: line_out += word + '\n'
            index += 1

        count[int(n[3])] += 1
        #
        data_file[int(n[3])].write(line_out)
        # with pandas
        #data_temp = pd.DataFrame(np.array(base_array), columns=base)
        #data_mass[int(n[3])] = pd.concat([data_mass[int(n[3])],data_temp],axis=1)

    for i in range(len(name_class)):
        data_file[i].close()
    
    for i in range(len(name_class)):
        print(name_class[i],'\t\t',count[i])

#write define col from row to file
def out(line,col,fout,config):
    n = line.split(',')
    line_out = ''
    index = 0
    empty_count = 1
    for i in col:
        ##########
        if (n[i] in config.flags["data_downloading"]["filter"]):
        #if(n[i] == "" or n[i] == 0):
            #empty_count = 1
            empty_count = 0
        if index == len(col)-1 :
            if(len(n[i].split('\n')) == 1):
                line_out += n[i] + '\n'
            else:
                line_out += n[i]
        else: line_out += n[i] + ','
        index += 1
    if(empty_count):
        fout.write(line_out)

#cut dublicate
def cut_cut(col,filein,fileout,config):
    fout = open(fileout,'w')
    gc1 = col[0]
    gc2 = col[1]
    #kill duplicate algorithm
    def f1():
        index_in,index_out = 0,0
        i=1
        z=0
        l=""
        for line in open(filein):
            n = line.split(',')
            #duplicate algorithm
            if (i>1):
                if (decn!=n[gc2]) or (ran!=n[gc1]):
                    ran=n[gc1]
                    decn=n[gc2]
                    if (z==1):
                        out(str(l),col,fout,config)
                        index_out+=1
                    l=str(line)
                    z=1                    
                else:
                    z+=1
            else:
                ran=n[gc1]
                decn=n[gc2]
                i=2
                z=1
                l=str(line)
            index_in+=1
        if (z==1):
            out(str(l),col,fout,config)
            index_out+=1
        return index_in, index_out
    #1-st of dublicate
    def f2():
        index_in,index_out = 0,0
        i=1
        l=""
        for line in open(filein):   
            n=line.split(',')    
            if (i>1):
                if (decn!=n[gc2]) or (ran!=n[gc1]):
                    ran=n[gc1]
                    decn=n[gc2]
                    out(str(line),col,fout,config)
                    index_out+=1
            else:
                ran=n[gc1]
                decn=n[gc2]
                i=2
                out(str(line),col,fout,config)
                index_out+=1
            index_in+=1
        return index_in, index_out
    
    if (config.flags["data_downloading"]["duplicate"]):
        return f2()
    else:
        return f1()

#request to VizieR X-match
def req(cat_name,name,fout,R=1):
    from astropy.table import Table
    from astropy.io.votable import from_table, writeto

    t = Table.read(f'{name}.csv', format = 'ascii.csv') 
    votable = from_table(t)
    writeto(votable, f'{name}.vot')
    #catalog_name = globals()[cat_name]

    import requests

    r = requests.post(
            'http://cdsxmatch.u-strasbg.fr/xmatch/api/v1/sync',
            data={'request': 'xmatch', 'distMaxArcsec': R, 'RESPONSEFORMAT': 'csv',
            'cat2': f'vizier:{cat_name}', 'colRA1': 'RA', 'colDec1': 'DEC'},
            files={'cat1': open(f'{name}.vot', 'r')})

    import os
    os.remove(f'{name}.vot')
    h = open(f'{fout}', 'w')
    h.write(r.text)
    h.close()

#division into pieces of define count
def slice(filename,count):
    import os
    foldername = filename.split('.')[0]
    if not os.path.isdir(foldername):
        os.mkdir(foldername)
    stat = []
    index = 0
    index_name = 1
    fout = open(f"{foldername}/0.csv","w")
    for line in open(filename):
        index+=1
        if(index % count == 1):
            fout.write('RA,DEC,z\n')
        fout.write(line)
        if(index // count == index_name):
            stat.append(count)
            fout_name = f"{index_name}.csv"
            fout.close()
            fout = open(f"{foldername}/{fout_name}","w")
            index_name+=1
    stat.append(index - (len(stat) + 1)*count)
    fout.close()
    #return foldername
    return stat

#download from VizieR
def download(catalogs_name,filepath,config):
    import os
    filepath_temp = filepath
    count_mass = np.zeros(len(catalogs_name)*2)
    columns = []
    for n, name in enumerate(catalogs_name):
        fin = filepath_temp.split(".")[0]
        temp = f"{fin}_{name}.csv"
        req(config.flags["data_downloading"]["catalogs"]["VizieR"][n],fin,temp,config.flags["data_downloading"]["radius"])
        
        if(n==0 and config.flags["data_downloading"]["remove"]["slice"]):
            os.remove(filepath_temp)
        if(not n==0 and config.flags["data_downloading"]["remove"]["catalogs_cross_cut_duplicate"][n-1]):
            os.remove(filepath_temp)
        
        filepath_temp = f"{temp.split('.')[0]}_cut.csv"
        
        count_mass[n*2],count_mass[n*2+1] = cut_cut(config.flags["data_downloading"]["catalogs"]["columns"][n],temp,filepath_temp,config)
        
        if(config.flags["data_downloading"]["remove"]["catalogs_cross_origin"][n]):
            os.remove(temp)
            #os.rename(filepath_temp,temp)
            #filepath_temp = temp
        columns.append(name)
        columns.append(f'{name}_cut')
    
    count_mass = pd.DataFrame(np.array(count_mass), columns=columns)

    return count_mass
        

def multi_thr_slice_download(catalogs_name,filename,config):
    MAX_WORKERS = config.flags["data_downloading"]["multi_thr"]["MAX_WORKERS"]
    WAIT_UNTIL_REPEAT_ACCESS = config.flags["data_downloading"]["multi_thr"]["WAIT_UNTIL_REPEAT_ACCESS"]
    NUM_URL_ACCESS_ATTEMPTS = config.flags["data_downloading"]["multi_thr"]["NUM_URL_ACCESS_ATTEMPTS"]
    import time
    import os   
    from concurrent.futures import ThreadPoolExecutor
    count_mass = []
    count_mass_temp = pd.DataFrame()
    attempts = 0
    while attempts < NUM_URL_ACCESS_ATTEMPTS:
        try:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                for name in os.listdir(filename):
                    count_mass_temp = executor.submit(download,catalogs_name,f"{filename}/{name}",config)
                    count_mass.append(count_mass_temp)
            break
        except:
            time.sleep(WAIT_UNTIL_REPEAT_ACCESS)
    
    count_mass = pd.DataFrame(np.array(count_mass), columns=count_mass_temp.columns.values)
    
    return count_mass

def slice_download(catalogs_name,filename,config):
    import os
    count_mass = pd.DataFrame()
    for name in os.listdir(filename):
        count_mass_temp = download(catalogs_name,f"{filename}/{name}",config)
        count_mass = pd.concat([count_mass,count_mass_temp],ignore_index=True)
    
    return count_mass

def unslice(filename_list,filename,fout,config):
    import os
    f = open(fout,"w")
    flag_2 = 1
    for name in filename_list:
        flag_1 = 1
        for line in open(f"{filename}/{name}"):
            if flag_1:
                if flag_2:
                    flag_2 = 0
                    f.write(line)
                flag_1 = 0
                continue
            if (line=='\n'):
                continue
            f.write(line)
        flag = 1
        
        if(config.flags["data_downloading"]["remove"]["catalogs_cross_cut_duplicate"][-1]):
            os.remove(f"{filename}/{name}")
    if(config.flags["data_downloading"]["remove"]["dir"]):
        os.rmdir(f"{filename}")
    
    f.close()

def class_download(name,path_sample,config):
    catalogs_names = config.flags["data_downloading"]["catalogs"]["name"]
    #differentiation origin catalog to slice
    stat_count = slice(f'{path_sample}/{name}_origin.csv', config.flags["data_downloading"]["slice_count"])
    stat_count = pd.DataFrame(np.array(stat_count), columns=['origin'])
    #cross-match feat. VizieR
    if(config.flags["data_downloading"]["multi_thr"]["work"]):
        count_cat = multi_thr_slice_download(['des_dr2'],f'{path_sample}/{name}',config)
        stat_count = pd.concat([stat_count,count_cat],axis=1)
    else:
        count_cat = slice_download(catalogs_names,f'{path_sample}/{name}',config)
        stat_count = pd.concat([stat_count,count_cat],axis=1)
    #
    filename_list = []
    for i in range(stat_count.shape[0]):
        line = str(i)
        for j in range(2,stat_count.shape[1],2):
            line += "_" + stat_count.columns.values[j]
        filename_list.append(f'{line}.csv')
    #
    unslice(filename_list,f'{path_sample}/{name}',f'{path_sample}/{name}.csv',config)
    #    
    if(config.flags["data_downloading"]["remove"]["origin"]):
        os.remove(f'{path_sample}/{name}_origin.csv')

    return stat_count