# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os

from main import general_path, base, flags

#sample_path = f'{general_path}/sample'

def diff_class(data_path,name_class,path_sample):
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
def out(line,col,fout):
    n = line.split(',')
    line_out = ''
    index = 0
    empty_count = 1
    for i in col:
        ##########
        if (n[i] in flags["data_downloading"]["filter"]):
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
def cut_cut(col,filein,fileout):
    fout = open(fileout,'w')
    gc1 = col[0]
    gc2 = col[1]
    #kill duplicate algorithm
    def f1():
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
                        out(str(l),col,fout)
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
        if (z==1):
            out(str(l),col,fout)
    #1-st of dublicate
    def f2():
        i=1
        l=""
        for line in open(filein):   
            n=line.split(',')    
            if (i>1):
                if (decn!=n[gc2]) or (ran!=n[gc1]):
                    ran=n[gc1]
                    decn=n[gc2]
                    out(str(line),col,fout)
            else:
                ran=n[gc1]
                decn=n[gc2]
                i=2
                out(str(line),col,fout)
    
    if (flags["data_downloading"]["duplicate"]):
        f2()
    else:
        f1()

#request to VizieR X-match
def req(cat_name,name,fout,R = flags["data_downloading"]["radius"]):
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
    index = 0
    index_name = 1
    fout = open(f"{foldername}/0.csv","w")
    for line in open(filename):
        index+=1
        if(index % count == 1):
            fout.write('RA,DEC,z\n')
        fout.write(line)
        if(index // count == index_name):
            fout_name = f"{index_name}.csv"
            fout.close()
            fout = open(f"{foldername}/{fout_name}","w")
            index_name+=1
    fout.close()
    #return foldername

#download from VizieR
def download(catalogs_name,filepath):
    import os
    for n, name in enumerate(catalogs_name):
        fin = filepath.split(".")[0]
        temp = f"{fin}_{name}.csv"
        req(flags["data_downloading"]["catalogs"]["VizieR"][n],fin,temp)
        os.remove(filepath)
        filepath = f"{temp.split('.')[0]}_cut.csv"
        cut_cut(flags["data_downloading"]["catalogs"]["columns"][n],temp,filepath)
        os.remove(temp)

def multi_thr_slice_download(catalogs_name,filename):
    MAX_WORKERS = flags["data_downloading"]["multi_thr"]["MAX_WORKERS"]
    WAIT_UNTIL_REPEAT_ACCESS = flags["data_downloading"]["multi_thr"]["WAIT_UNTIL_REPEAT_ACCESS"]
    NUM_URL_ACCESS_ATTEMPTS = flags["data_downloading"]["multi_thr"]["NUM_URL_ACCESS_ATTEMPTS"]
    import time
    import os   
    from concurrent.futures import ThreadPoolExecutor
    attempts = 0
    while attempts < NUM_URL_ACCESS_ATTEMPTS:
        try:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                for name in os.listdir(filename):
                    executor.submit(download,catalogs_name,f"{filename}/{name}")
            break
        except:
            time.sleep(WAIT_UNTIL_REPEAT_ACCESS)

def slice_download(catalogs_name,filename):
    import os
    for name in os.listdir(filename):
        download(catalogs_name,f"{filename}/{name}")

def unslice(filename,fout):
    import os
    f = open(fout,"w")
    flag_2 = 1
    for name in os.listdir(filename):
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
        os.remove(f"{filename}/{name}")
    os.rmdir(f"{filename}")
    f.close()
'''
def dir(save_path,name):
    dir_name = f"{save_path}/{name}"
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
'''
def class_download(name,path_sample):
    catalogs_names = flags["data_downloading"]["catalogs"]["name"]
    slice(f'{path_sample}/{name}_origin.csv', flags["data_downloading"]["slice_count"])
    if(flags["data_downloading"]["multi_thr"]["work"]):
        multi_thr_slice_download(['des_dr2'],f'{path_sample}/{name}')
    else:
        slice_download(catalogs_names,f'{path_sample}/{name}')
    unslice(name,f'{path_sample}/{name}.csv')
    #os.remove(f'{name}.csv')
