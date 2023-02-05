# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

save_path = r'C:\Users\lrikozavr\github\extragalactic_object_locate_with_ml'
filepath = f'{save_path}\exit.sort'
'''
#create file for class obj exgal and star
f_exgal = open('exgal.csv','w')
f_star = open('star.csv','w')
#write header for VizieR
f_exgal.write('RA,DEC,z\n')
f_star.write('RA,DEC,z\n')

#extract exgal and star obj from file exit.sort
count_exgal = 0
count_star = 0
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
    if not int(n[3]) == 0:
        count_exgal += 1
        f_exgal.write(line_out)
    else: 
        f_star.write(line_out)
        count_star += 1
print(count_star)
print(count_exgal)
'''
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
    for i in col:
        if index == len(col)-1 :
            line_out += n[i] + '\n'
        else: line_out += n[i] + ','
        index += 1
    fout.write(line_out)

#kill duplicate algorithm
def cut_cut(col,filein,fileout):
    fout = open(fileout,'w')
    head = 1
    i=1
    z=0
    l=""
    gc1 = col[0]
    gc2 = col[1]
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

#VizieR path
allwise = 'II/328/allwise'
gaiadr3 = 'I/350/gaiaedr3'
des_dr2 = 'II/371/des_dr2'
#usful column index for each catalogs
col_allwise = [1,2,3,5,6,10,11,17,18]
#col_gaiadr3 = [1,2,3,4,5,6,7,8,9,16,18,41,44,46,58,59,60]
col_gaiadr3 = [1,2,3,6,8,7,9,41,58,44,59,46,60]
col_des_dr2 = [1,2,3,11,12,18,19,20,21,22,23,24,25,26,27]

#request to VizieR X-match
def req(cat_name,name,fout):
    from astropy.table import Table
    from astropy.io.votable import from_table, writeto

    t = Table.read(f'{name}.csv', format = 'ascii.csv') 
    votable = from_table(t)
    writeto(votable, f'{name}.vot')

    import requests

    r = requests.post(
            'http://cdsxmatch.u-strasbg.fr/xmatch/api/v1/sync',
            data={'request': 'xmatch', 'distMaxArcsec': 3, 'RESPONSEFORMAT': 'csv',
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
    foldername = filename.split(',')[0]
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
    return foldername

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
            f.write(line)
        flag = 1
        os.remove(f"{filename}/{name}")
    os.remove(f"{filename}")
    f.close()

req(des_dr2,'ml/qso_sample','des_qso.csv')

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