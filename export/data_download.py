# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os


#sample_path = f'{general_path}/sample'
def get_col_list(columns,config):
    col = []
    #print(columns)
    for column in config.base:
        col.append(columns.index(column))
    for column in config.features["data"]:
        try:
            col.append(columns.index(column))
        except:
            continue
    return col

def diff_class(config):
    #create file for class obj
    #data_mass = []
    #for i in range(len(name_class)):
    #    data_mass.append(pd.DataFrame())
    #

    #extract exgal and star obj from file exit.sort
    count = np.zeros(len(config.name_class))

    f = open(config.data_path,'r')
    first_line = f.readline().strip('\n').split(",")
    base_index_array = get_col_list(first_line, config)
    name_class_column_index = first_line.index(config.name_class_column)

    data_file = []
    for i in range(len(config.name_class)):
        if(len(base_index_array) == len(config.base)):
            f = open(f'{config.path_sample}/{config.name_class[i]}_origin.csv','w')
        else:
            f = open(f'{config.path_sample}/{config.name_class[i]}.csv','w')
        f.write(','.join([first_line[base_index_array[i]] for i in range(len(base_index_array))])+"\n")
        data_file.append(f)

    for line in f:
        n = line.strip('\n').split(',')
        base_array = [n[base_index_array[i]] for i in range(len(base_index_array))]
        #if(len(n[4].split('_')) == 1 and (n[4] == 'lamost' or n[4] == 'sdss')):
        index = 0 
        line_out = ''
        for word in base_array:
            if(not index == len(base_index_array) - 1):
                line_out += word + ','
            else: line_out += word + '\n'
            index += 1

        count[int(n[name_class_column_index])] += 1
        #
        data_file[int(n[name_class_column_index])].write(line_out)
        # with pandas
        #data_temp = pd.DataFrame(np.array(base_array), columns=base)
        #data_mass[int(n[3])] = pd.concat([data_mass[int(n[3])],data_temp],axis=1)

    for i in range(len(config.name_class)):
        data_file[i].close()
    
    for i in range(len(config.name_class)):
        print(config.name_class[i],'\t\t',count[i])

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
def cut_cut(filein,fileout,config):
    fout = open(fileout,'w')
    #
    f = open(filein,'r')
    col = get_col_list(f.readline().strip('\n').split(','),config)
    #print(col)
    #
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
def req(cat_name,name,fout,config):
    from astropy.table import Table
    from astropy.io.votable import from_table, writeto

    t = Table.read(f'{name}.csv', format = 'ascii.csv') 
    votable = from_table(t)
    writeto(votable, f'{name}.vot')
    #catalog_name = globals()[cat_name]

    import requests
    import time

    WAIT_UNTIL_REPEAT_ACCESS = config.flags["data_downloading"]["multi_thr"]["WAIT_UNTIL_REPEAT_ACCESS"]
    NUM_URL_ACCESS_ATTEMPTS = config.flags["data_downloading"]["multi_thr"]["NUM_URL_ACCESS_ATTEMPTS"]

    attempts = 0
    while attempts < NUM_URL_ACCESS_ATTEMPTS:
        try:
            r = requests.post('http://cdsxmatch.u-strasbg.fr/xmatch/api/v1/sync',
                            data={  'request': 'xmatch', 
                                    'distMaxArcsec': config.flags["data_downloading"]["radius"], 
                                    'RESPONSEFORMAT': 'csv',
                                    'cat2': f'vizier:{cat_name}', 'colRA1': 'RA', 'colDec1': 'DEC'},
                                    files={'cat1': open(f'{name}.vot', 'r')})
            if(r.ok):
                os.remove(f'{name}.vot')
                h = open(f'{fout}', 'w')
                h.write(r.text)
                h.close()
                break
            else:
                raise Exception(r.raise_for_status())
        except:
            time.sleep(WAIT_UNTIL_REPEAT_ACCESS)
            attempts += 1
    
    if(attempts == NUM_URL_ACCESS_ATTEMPTS):
        raise Exception('')
    
    

#division into pieces of define count
def slice(filename,foldername,count,base):
    print("cut to slice start for ",filename)
    import os
    #foldername = filename.split('_')[0]
    if not os.path.isdir(foldername):
        os.mkdir(foldername)
    filelist = []
    stat = []
    index = 0
    index_name = 1
    fout = open(f"{foldername}/0.csv","w")
    fin = open(filename,'r')
    print('Origin catalog columns:\t',fin.readline().strip('\n'))
    for line in fin:
        index+=1
        if(index % count == 1):
            fout.write(','.join(base)+'\n')
        fout.write(line)
        if(index // count == index_name):
            filelist.append(f"{index_name-1}.csv")
            stat.append(count)
            fout_name = f"{index_name}.csv"
            fout.close()
            fout = open(f"{foldername}/{fout_name}","w")
            index_name+=1
    stat.append(index - len(stat)*count)
    filelist.append(f"{index_name-1}.csv")
    fout.close()
    print("cut to slice finish with ",index_name-1," slices")
    #return foldername
    return stat, filelist

#download from VizieR
def download(catalogs_name,filepath,config):
    import os
    filepath_temp = filepath
    count_mass = np.zeros(len(catalogs_name)*2)
    columns = []
    for n, name in enumerate(catalogs_name):
        fin = filepath_temp.split(".")[0]
        temp = f"{fin}_{name}.csv"
        req(config.flags["data_downloading"]["catalogs"]["VizieR"][n],fin,temp,config)
        
        if(n==0 and config.flags["data_downloading"]["remove"]["slice"]):
            os.remove(filepath_temp)
        if(not n==0 and config.flags["data_downloading"]["remove"]["catalogs_cross_cut_duplicate"][n-1]):
            os.remove(filepath_temp)
        
        filepath_temp = f"{temp.split('.')[0]}_cut.csv"
        
        #config.flags["data_downloading"]["catalogs"]["columns"][n],
        count_mass[n*2],count_mass[n*2+1] = cut_cut(temp,filepath_temp,config)
        
        if(config.flags["data_downloading"]["remove"]["catalog_cross_origin"][n]):
            os.remove(temp)
            #os.rename(filepath_temp,temp)
            #filepath_temp = temp
        columns.extend([name,f'{name}_cut'])
    #print(count_mass[0])
    #print(columns)
    count_mass = pd.DataFrame([np.array(count_mass)], columns=columns)
    #print(count_mass)

    return count_mass
        

def multi_thr_slice_download(catalogs_name,filename,filelist,config):
    MAX_WORKERS = config.flags["data_downloading"]["multi_thr"]["MAX_WORKERS"]
      
    from concurrent.futures import ThreadPoolExecutor
    count_mass = []
    attempts = 0    
    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for name in filelist:
                count_mass_temp = executor.submit(download,catalogs_name,f"{filename}/{name}",config)
                #print(count_mass_temp)
                count_mass.append(count_mass_temp)
    except:
        raise Exception("Except: download part have issue. \nCheck your Ethernet conection, \nor config->flags->data_downloading variable, \nor origin catalog purity")
        
        #attempts += 1
    
    stat = pd.DataFrame()
    for i in range(len(count_mass)):
        stat = pd.concat([stat,count_mass[i].result()],ignore_index=True)
    
    return stat

def slice_download(catalogs_name,filename,filelist,config):
    import os
    count_mass = pd.DataFrame()
    for name in filelist:
        count_mass_temp = download(catalogs_name,f"{filename}/{name}",config)
        count_mass = pd.concat([count_mass,count_mass_temp],ignore_index=True)
    
    return count_mass

def unslice(filename_list,filename,fout,config):
    import os
    f = open(fout,"w")
    for n, name in enumerate(filename_list):
        f_slice = open(f"{filename}/{name}",'r')
        first_line = f_slice.readline()
        if(n==0):
            f.write(first_line)
        for line in f_slice:
            if (line=='\n'):
                continue
            f.write(line)
        f_slice.close()
        
        if(config.flags["data_downloading"]["remove"]["catalogs_cross_cut_duplicate"][-1]):
            os.remove(f"{filename}/{name}")
    
    if(config.flags["data_downloading"]["remove"]["dir"]):
        import shutil
        shutil.rmtree(filename)
        #os.rmdir(f"{filename}")
    
    f.close()

def class_download(name,path_sample,config):
    
    catalogs_names = config.flags["data_downloading"]["catalogs"]["name"]
    #differentiation origin catalog to slice
    stat_count, filelist = slice(f'{path_sample}/{name}_origin.csv',f'{path_sample}/{name}', config.flags["data_downloading"]["slice_count"],config.base)
    stat_count = pd.DataFrame(np.array(stat_count), columns=['origin'])
    #cross-match feat. VizieR
    if(config.flags["data_downloading"]["multi_thr"]["work"]):
        count_cat = multi_thr_slice_download(catalogs_names,f'{path_sample}/{name}',filelist,config)
        stat_count = pd.concat([stat_count,count_cat],axis=1)
    else:
        count_cat = slice_download(catalogs_names,f'{path_sample}/{name}',filelist,config)
        stat_count = pd.concat([stat_count,count_cat],axis=1)
    #
    #stat_count = pd.read_csv(f'{config.path_stat}/{name}_slice.log', header=0, sep=',')
    #
    filename_list = []
    for i in range(stat_count.shape[0]):
        line = str(i)
        for j in range(2,stat_count.shape[1],2):
            line += "_" + stat_count.columns.values[j]
        filename_list.append(f'{line}.csv')
    #
    print(filename_list)
    unslice(filename_list,f'{path_sample}/{name}',f'{path_sample}/{name}.csv',config)
    #    
    if(config.flags["data_downloading"]["remove"]["origin"]):
        os.remove(f'{path_sample}/{name}_origin.csv')

    return stat_count