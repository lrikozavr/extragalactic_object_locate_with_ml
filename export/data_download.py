# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os

from types import SimpleNamespace
import requests
import time

#sample_path = f'{general_path}/sample'
def get_col_list(columns, base_columns, features):
    col = []
    #print(columns)
    for column in base_columns:
        col.append(columns.index(column))
    if(type(features) is type(dict())):
        for key in features.keys():
            for column in [key]:
                try:
                    col.append(columns.index(column))
                except:
                    continue
    else:                
        for column in features:
            try:
                col.append(columns.index(column))
            except:
                continue
    return col

#write define col from row to file
def out(line, col, fout, gate, filter):
    '''
    gate = 1 \n
    filter = [" ", "---"]
    '''
    n = line.split(',')
    line_out = ''
    index = 0
    empty_count = 1
    #gate
    try:
        if (float(n[0]) > gate):
            empty_count = 0
    except:
        print()
    for i in col:
        if (n[i] in filter):
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

class Download:

    flag = SimpleNamespace()
    value = SimpleNamespace()
    path = SimpleNamespace()
    columns = SimpleNamespace()

    data = SimpleNamespace()

    def __init__(self, config_spec = [], config_base = []):
        if(config_spec == []):
            self.flag.duplicate = False
            self.flag.class_diff = False
            self.flag.remove = dict(origin=False,
                                    slice=False, 
                                    catalog_cross_origin=[False, False],
                                    catalogs_cross_cut_duplicate=[False, False],
                                    dir=False)
            
            self.value.radius = 1
            self.value.gate = 1
            self.value.filter = [" ", 0]
            self.value.slice_count = 100000
            self.value.multi_thr = dict(MAX_WORKERS = 3,
                                        WAIT_UNTIL_REPEAT_ACCESS = 3,
                                        NUM_URL_ACCESS_ATTEMPTS = 10)
            self.value.catalogs = dict(name = ["catwise","gaiadr3"],
                                        VizieR = ["II/365/catwise","I/355/gaiadr3"])
        else:        
            self.flag.duplicate = config_spec["duplicate"]
            self.flag.class_diff = config_spec["class_diff"]
            self.flag.remove = config_spec["remove"]
            
            self.value.radius = config_spec["radius"]
            self.value.gate = config_spec["gate"]
            self.value.filter = config_spec["filter"]
            self.value.slice_count = config_spec["slice_count"]
            self.value.multi_thr = config_spec["multi_thr"]
            self.value.catalogs = config_spec["catalogs"]
        
        if config_base == []:
            self.columns.name_class = []
            self.columns.name_class_column = []
            self.columns.base = []
            self.columns.features = []
        
            self.path.data_path = ""
            self.path.path_sample = ""
        else:
            self.columns.name_class = config_base.name_class
            self.columns.name_class_column = config_base.name_class_column
            self.columns.base = config_base.base
            self.columns.features = config_base.features
            
            self.path.data_path = config_base.data_path
            self.path.path_sample = config_base.path_sample

    


    def diff_class(self):
        #create file for class obj
        #data_mass = []
        #for i in range(len(name_class)):
        #    data_mass.append(pd.DataFrame())
        #

        #extract exgal and star obj from file exit.sort
        count = np.zeros(len(self.columns.name_class))

        fin = open(self.path.data_path,'r')
        first_line = fin.readline().strip('\n').split(",")
        base_index_array = get_col_list(first_line, self.columns.base, self.columns.features["data"])
        name_class_column_index = first_line.index(self.columns.name_class_column)

        data_file = []
        for i in range(len(self.columns.name_class)):
            if(len(base_index_array) == len(self.columns.base)):
                f = open(f'{self.path.path_sample}/{self.columns.name_class[i]}_origin.csv','w')
            else:
                f = open(f'{self.path.path_sample}/{self.columns.name_class[i]}.csv','w')
            f.write(','.join([first_line[base_index_array[i]] for i in range(len(base_index_array))])+"\n")
            data_file.append(f)

        for line in fin:
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

        for i in range(len(self.columns.name_class)):
            data_file[i].close()
        
        for i in range(len(self.columns.name_class)):
            print(self.columns.name_class[i],'\t\t',count[i])



    #cut dublicate
    def cut_cut(self,filein,fileout):
        fout = open(fileout,'w')
        #
        f = open(filein,'r')
        col = get_col_list(f.readline().strip('\n').split(','), self.columns.base, self.columns["features"]["data"])
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
                            out(str(l),col,fout,self.value["gate"],self.value["filter"])
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
                out(str(l),col,fout,self.value["gate"],self.value["filter"])
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
                        out(str(line),col,fout,self.value["gate"],self.value["filter"])
                        index_out+=1
                else:
                    ran=n[gc1]
                    decn=n[gc2]
                    i=2
                    out(str(line),col,fout,self.value["gate"],self.value["filter"])
                    index_out+=1
                index_in+=1
            return index_in, index_out
        
        if (self.flag["duplicate"]):
            return f2()
        else:
            return f1()

    #request to VizieR X-match
    def req(self,cat_name,name,fout):
        from astropy.table import Table
        from astropy.io.votable import from_table, writeto

        t = Table.read(f'{name}.csv', format = 'ascii.csv') 
        votable = from_table(t)
        writeto(votable, f'{name}.vot')
        #catalog_name = globals()[cat_name]

        WAIT_UNTIL_REPEAT_ACCESS = self.value["multi_thr"]["WAIT_UNTIL_REPEAT_ACCESS"]
        NUM_URL_ACCESS_ATTEMPTS = self.value["multi_thr"]["NUM_URL_ACCESS_ATTEMPTS"]

        attempts = 0
        while attempts < NUM_URL_ACCESS_ATTEMPTS:
            try:
                r = requests.post('http://cdsxmatch.u-strasbg.fr/xmatch/api/v1/sync',
                                data={  'request': 'xmatch', 
                                        'distMaxArcsec': self.value["radius"], 
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



    #download from VizieR
    def download(self,catalogs_name,filepath):
        import os
        filepath_temp = filepath
        count_mass = np.zeros(len(catalogs_name)*2)
        columns = []
        for n, name in enumerate(catalogs_name):
            fin = filepath_temp.split(".")[0]
            temp = f"{fin}_{name}.csv"
            self.req(self.value["catalogs"]["VizieR"][n],fin,temp)
            
            if(n==0 and self.flag["remove"]["slice"]):
                os.remove(filepath_temp)
            if(not n==0 and self.flag["remove"]["catalogs_cross_cut_duplicate"][n-1]):
                os.remove(filepath_temp)
            
            filepath_temp = f"{temp.split('.')[0]}_cut.csv"
            
            #config.flags["data_downloading"]["catalogs"]["columns"][n],
            count_mass[n*2],count_mass[n*2+1] = self.cut_cut(temp,filepath_temp)
            
            if(self.flag["remove"]["catalog_cross_origin"][n]):
                os.remove(temp)
                #os.rename(filepath_temp,temp)
                #filepath_temp = temp
            columns.extend([name,f'{name}_cut'])
        #print(count_mass[0])
        #print(columns)
        count_mass = pd.DataFrame([np.array(count_mass)], columns=columns)
        #print(count_mass)

        return count_mass
        

    def multi_thr_slice_download(self,catalogs_name,filename,filelist):
        MAX_WORKERS = self.value["multi_thr"]["MAX_WORKERS"]
        
        from concurrent.futures import ThreadPoolExecutor
        count_mass = []
        #attempts = 0    
        try:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                for name in filelist:
                    count_mass_temp = executor.submit(self.download,catalogs_name,f"{filename}/{name}")
                    #print(count_mass_temp)
                    count_mass.append(count_mass_temp)
        except:
            raise Exception("Except: download part have issue. \nCheck your Ethernet conection, \nor config->flags->data_downloading variable, \nor origin catalog purity")
            
            #attempts += 1
        
        stat = pd.DataFrame()
        for i in range(len(count_mass)):
            stat = pd.concat([stat,count_mass[i].result()],ignore_index=True)
        
        return stat

    def slice_download(self,catalogs_name,filename,filelist):
        import os
        count_mass = pd.DataFrame()
        for name in filelist:
            count_mass_temp = self.download(catalogs_name,f"{filename}/{name}")
            count_mass = pd.concat([count_mass,count_mass_temp],ignore_index=True)
        
        return count_mass

    def unslice(self,filename_list,filename,fout):
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
            
            if(self.flag["remove"]["catalogs_cross_cut_duplicate"][-1]):
                os.remove(f"{filename}/{name}")
        
        if(self.flag["remove"]["dir"]):
            import shutil
            shutil.rmtree(filename)
            #os.rmdir(f"{filename}")
        
        f.close()

    def class_download(self,name,path_sample):
        
        catalogs_names = self.value["catalogs"]["name"]
        #differentiation origin catalog to slice
        stat_count, filelist = slice(f'{path_sample}/{name}_origin.csv',f'{path_sample}/{name}', self.value["slice_count"], self.columns.base)
        stat_count = pd.DataFrame(np.array(stat_count), columns=['origin'])
        #cross-match feat. VizieR
        if(self.value["multi_thr"]["work"]):
            count_cat = self.multi_thr_slice_download(catalogs_names,f'{path_sample}/{name}',filelist)
            stat_count = pd.concat([stat_count,count_cat],axis=1)
        else:
            count_cat = self.slice_download(catalogs_names,f'{path_sample}/{name}',filelist)
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
        self.unslice(filename_list,f'{path_sample}/{name}',f'{path_sample}/{name}.csv')
        #    
        if(self.flag["remove"]["origin"]):
            os.remove(f'{path_sample}/{name}_origin.csv')

        return stat_count

    def download_catalog(self,filename,url,cat):
        response = requests.get(url)
        data = response.content
        print(data)
        open(filename,'wb').write(data)
        data_str = str(data).split('\n')
        row_count, col_count = len(data_str), len(cat.keys())
        mass = np.array((row_count, col_count))
        for i in range(row_count):
            data_byte = bytes(data_str[i], 'utf8')
            for n, name in enumerate(cat.keys()):
                st_fn = cat.get(name)
                mass[i,n] = str(data_byte[st_fn[0]:st_fn[1]]) 
        data_done = pd.DataFrame(mass,columns=cat.keys())
        return data_done