# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
save_path = '/home/lrikozavr/catalogs/elewen'
#save_path = r'C:\Users\lrikozavr\github\extragalactic_object_locate_with_ml'

#filepath = f'{save_path}/exit.sort'
filepath = f'{save_path}/exit_single_notlamost.csv'


#general_path = '/home/lrikozavr/ML_work/des_pro'
general_path = '/home/lrikozavr/ML_work/allwise_gaiadr3'
#sample_path = f'{general_path}/sample_3'
sample_path = f'{general_path}/sample'

#create file for class obj exgal and star

f_exgal = open('qso.csv','w')
f_star = open('star.csv','w')
f_gal = open('gal.csv','w')

#write header for VizieR
#f_exgal.write('RA,DEC,z\n')
#f_star.write('RA,DEC,z\n')

#extract exgal and star obj from file exit.sort
count_qso = 0
count_star = 0
count_gal = 0
for line in open(filepath):
    n = line.split(',')
    #if(len(n[4].split('_')) == 1 and (n[4] == 'lamost' or n[4] == 'sdss')):
    index = 0 
    line_out = ''
    for i in n:
        if(index < 3):
            if(not index == 2):
                line_out += n[index] + ','
            else: line_out += n[index] + '\n'
        else:
            break
        index += 1
    if(int(n[3]) == 1):
        count_qso += 1
        f_exgal.write(line_out)
    elif(int(n[3]) == 2):
        count_gal += 1
        f_gal.write(line_out)
    else: 
        f_star.write(line_out)
        count_star += 1
print(count_star)
print(count_qso)
print(count_gal)

#koef_star = 1e5 / index_star
#koef_exgal = 1e5 / index_exgal
#cut out define % line from file 
def cut_out(filepath,filename,koef):
    f = open(filename,'w')
    f.write('RA,DEC,z\n')
    count = 0 
    t = 0
    for i in open(filepath):
        if (np.random.rand(1) < koef and t > 0):
            f.write(i)
            count += 1
        t = 1
    print(count)
#cut_out('qso.csv','qso_cut.csv',koef_qso)
#cut_out('star.csv','star_cut.csv',koef_star)

#write define col from row to file
def out(line,col,fout):
    n = line.split(',')
    line_out = ''
    index = 0
    empty_count = 1
    for i in col:
        if(n[i] == ""):
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
    decn,ran = '',''
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
    f1()
#VizieR path
allwise = 'II/328/allwise'
gaiadr3 = 'I/350/gaiaedr3'
des_dr2 = 'II/371/des_dr2'
#usful column index for each catalogs
col_allwise = [1,2,3,5,6,10,11,12,17,18,19]
#col_gaiadr3 = [1,2,3,4,5,6,7,8,9,16,18,41,44,46,58,59,60]
col_gaiadr3 = [1,2,3,6,9,7,10,8,11,43,60,46,61,48,62]
col_des_dr2 = [1,2,3,11,12,18,23,19,24,20,25,21,26,22,27]

#request to VizieR X-match
def req(cat_name,name,fout,R = 1):
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
            'cat2': f'vizier:{globals()[cat_name]}', 'colRA1': 'RA', 'colDec1': 'DEC'},
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
    #return foldername

#download from VizieR
def download(catalogs_name,filepath):
    import os
    for name in catalogs_name:
        fin = filepath.split(".")[0]
        temp = f"{fin}_{name}.csv"
        req(name,fin,temp)
        os.remove(filepath)
        filepath = f"{temp.split('.')[0]}_cut.csv"
        cut_cut(globals()[f'col_{name}'],temp,filepath)
        os.remove(temp)

def multi_thr_slice_download(catalogs_name,filename):
    MAX_WORKERS = 6
    WAIT_UNTIL_REPEAT_ACCESS = 3
    NUM_URL_ACCESS_ATTEMPTS = 4
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
def clas(name,catalogs_names,sample_path):
    slice(f'{name}.csv',1e5)
        #multi_thr_slice_download(['des_dr2'],name)
    slice_download(catalogs_names,name)
    unslice(name,f'{sample_path}/{name}_without_lamost.csv')
    os.remove(f'{name}.csv')

import os

if not os.path.isdir(sample_path):
    os.mkdir(sample_path)

#clas('qso',['allwise','gaiadr3'],sample_path)
clas('star',['allwise','gaiadr3'],sample_path)
#clas('gal',['allwise','gaiadr3'],sample_path)

#req(des_dr2,'ml/qso_sample','des_qso.csv')



'''
def flag_z(name):
    fout = open(f'ml/{name}_sample_1.csv','w')
    for li in open(f'ml/{name}_sample.csv'):
        n = li.split(',')
        count = 0
        for i in range(len(n)):
            if(str(n[i]) == ''):
                count+=1
        if count > 0:
            continue
        else: fout.write(li)

flag_z('qso')
flag_z('star')
'''