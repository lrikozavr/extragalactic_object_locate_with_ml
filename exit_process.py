# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

save_path = r'C:\Users\lrikozavr\github\extragalactic_object_locate_with_ml'
filepath = f'{save_path}\exit.sort'
'''
f_qso = open('qso.csv','w')
f_star = open('star.csv','w')
f_qso.write('RA,DEC,z\n')
f_star.write('RA,DEC,z\n')

index_qso = 0
index_star = 0
for line in open(filepath):
    n = line.split(',')
    if(len(n[4].split('_')) == 1 and (n[4] == 'lamost' or n[4] == 'sdss')):
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
            index_qso += 1
            f_qso.write(line_out)
        else: 
            f_star.write(line_out)
            index_star += 1

print(index_star)
print(index_qso)
koef_star = 1e5 / index_star
koef_qso = 1e5 / index_qso

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

cut_out('qso.csv','qso_cut.csv',koef_qso)
cut_out('star.csv','star_cut.csv',koef_star)
'''
'''
allwise = 'II/328/allwise'
gaiadr3 = 'I/350/gaiaedr3'

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

    h = open(f'{fout}', 'w')
    h.write(r.text)
    h.close()

def slice(filename,count):
    import os
    foldername = filename.split(',')[0]
    if not os.path.isdir(foldername):
        os.mkdir(foldername)
    
    index = 0
    index_name = 1
    fout = open(f"{foldername}/0.csv","w")
    for line open(filename):
        index+=1
        fout.write(line)
        if(index // count == index_name):
            fout_name = f"{index_name}.csv"
            fout.close()
            fout = open(f"{foldername}/{fout_name}","w")
            index_name+=1
    return foldername

#req(gaiadr3,'qso','gaia_qso.csv')

col_allwise = [1,2,3,5,6,10,11,17,18]
#col_gaiadr3 = [1,2,3,4,5,6,7,8,9,16,18,41,44,46,58,59,60]
col_gaiadr3 = [1,2,3,6,8,7,9,41,58,44,59,46,60]


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

def cut_cut(col,filein,fileout):
    fout = open(fileout,'w')
    head = 1
    for line in open(filein):
        n = line.split(',')
        #duplicate algorithm
        gc1 = col[0]
        gc2 = col[1]
        i=1
        k=0
        z=0
        l=""    
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

def sample_req(name):
    req(allwise,f'{name}_cut',f'allwise_{name}.csv')
    cut_cut(col_allwise,f'allwise_{name}.csv',f'allwise_{name}_d.csv')
    req(gaiadr3,f'allwise_{name}_d',f'gaiadr3_allwise_{name}.csv')
    cut_cut(col_gaiadr3,f'gaiadr3_allwise_{name}.csv',f'{name}_sample.csv')

sample_req('qso')
sample_req('star')
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