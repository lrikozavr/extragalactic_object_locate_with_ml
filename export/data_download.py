# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import requests
import time

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.io.votable import from_table, writeto

import math

def loading_progress_bar(percent):
    bar_length = 50
    filled_length = int(percent * bar_length)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    print(f'\rProgress: |{bar}| {percent*100:.2f}% ', end='')

def cross_match(table_1,table_2,radius: float = 1.,flag: int = 1):

    def compare(cor1,cor2):
        return
    
    table_2_count = len(table_2)
    table_1_count = len(table_1)

    table_2_flag = np.zeros(table_2_count)

    table_1_index = np.zeros(table_1_count)

    flag_check = False

    j_start, j_finish = 0, table_2_count
    for index_1 in range(table_1):
        ra_1, dec_1 = table_1[index_1][0], table_1[index_1][1]
        
        delta_min, delta_min_jindex = 1e5, -1
        for jindex_2 in range(j_start,j_finish,1):
            ra_2, dec_2 = table_2[jindex_2][0], table_2[jindex_2][1]
            if(dec_1 + radius > dec_2 and dec_1 - radius < dec_2):
                if(flag_check == False):
                    j_start = jindex_2
                    flag_check = True
                if(ra_1 + radius > ra_2 and ra_1 - radius < ra_2):
                    if(table_2_flag[jindex_2] == 0):
                        delta = pow(pow(ra_1-ra_2,2) + pow(dec_1-dec_2,2),2)/math.cos(dec_1)
                        if(delta < radius):
                            if(delta < delta_min):
                                delta_min = delta
                                delta_min_jindex = jindex_2
                    
            elif(dec_1 + radius <= dec_2):
                flag_check = False
                break

        if(not delta_min_jindex == -1):
            table_2_flag[delta_min_jindex] = 1
            table_1_index[index_1] = delta_min_jindex

    '''    
    c = SkyCoord(ra=table_1["RA"]*u.degree, dec=table_1["DEC"]*u.degree)
    catalog = SkyCoord(ra=table_1["RA"]*u.degree, dec=table_2["DEC"]*u.degree)
    idx, d2d, d3d = c.match_to_catalog_sky(catalog)
    return catalog[idx]
    '''

def cut_cut_pd(data: pd.DataFrame, col1: list, col2: list):
    '''
    Видаляє дублікати по колонкам ``col1`` та ``col2``

    Input:
    ------
        ``data`` --- DataFrame з даними\n
        ``col1`` --- Колонки які характеризують дублікати\n
        ``col2`` --- Див. ``col1``\n
    
    Output:
    -------
        DataFrame без дублікатів
    
    '''

    data_duplicate_1 = data.duplicated(subset=col1, keep=False)
    data_duplicate_2 = data.duplicated(subset=col2, keep=False)

    index = data[((data_duplicate_1 == True | data_duplicate_2 == True))].index

    return data.drop(axis=0, index=index).reset_index(drop=True)

def download_data_style_catalog(filename: str, url: str, cat: dict):
    """
    Завантажує каталог за посиланням ``url``, зберігає у файлі ``filename`` та 
    виводить значення його у вигляді таблиці у відповідності до параметрів
    зазначених у словнику ``cat``.

    >>> cat = dict(RA = [1,5], DEC = [6,10])
    
    Input:
    ------
        ``filename`` --- назва/шлях файлу збереження \n
        ``url`` --- посилання на каталог \n
        ``cat`` --- словник з параметрами виділення колонок

    Output:
    -------
        ``result`` --- каталог з певними колонками

    """        
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
    return pd.DataFrame(mass,columns=cat.keys())

#sample_path = f'{general_path}/sample'
def get_col_list(columns: list, base_columns: list, features: dict | list):
    '''
    Створює масив індексів знаходження значень списків ``base_columns`` та ``features`` у списку ``columns``

    Input:
    ------
        ``columns``         --- список значень\n
        ``base_columns``    --- список\n
        ``features``        --- список \n
    
    Output:
    -------
        ``col`` --- список індексів\n

    '''
    col = list()
    #print(columns)
    for column in base_columns:
        col.append(columns.index(column))
    if(type(features) is type(dict())):
        for key in features.keys():
            for column in features[key]:
                try:
                    col.extend([i for i, value in enumerate(columns) if value == column])
                except:
                    continue
    else:                
        for column in features:
            try:
                col.extend([i for i, value in enumerate(columns) if value == column])
            except:
                continue
    return col

#write define col from row to file
def out(line: str, col: list, fout, gate: int, filter: list):
    """
    Записує значення колонок строки ``line`` з індексами ``col`` у файл ``fout`` 

    Фільтрація по: \n
        величині відстані між ототожненими об'єктами \n
            ``gate <= 1`` \n
        значенням кожної з колонок \n
            ``filter = [" ", "---"]`` \n
    Вивід відбувається у файл формату *.csv \n

    Input:
    ------
        ``line``    --- рядок у форматі .csv\n
        ``col``     --- список індексів\n
        ``fout``    --- файл виводу (формат *.csv)\n
        ``gate``    --- порогове значення відстані між ототожненнями\n
        ``filter``  --- колонки які співпадають зі значенням викидуються\n

    Output:
    -------
        EMPTY

    """
    #additional condition --- ','
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
        return 1
    return 0

#division into pieces of define count
def slice(filename: str, foldername: str, count: int):
    '''
    Ділить файл під назвою ``filename`` на певну кількість файлів з кількістю рядків ``count``\n
    та зберігає у папці ``foldername``. Результуючі файли мають нумерацію від 0 до ... у форматі *.csv
    
    Input:
    ------
        ``filename``    --- назва файлу формату *.csv\n
        ``foldername``  --- назва папки (якщо її не існує створить її)\n
        ``count``       --- кількість рядків у кожному з фалів списку\n
    
    Output:
    -------
        ``stat``        --- список значень кількості рядків у кожному з файлів\n
       ``filelist``    --- список назв файлів

    '''
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
    base = fin.readline().strip('\n').split(",")
    print('Origin catalog columns:\t',base)
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
    print("total count of lines is ",index)
    #return foldername
    return stat, filelist

class Download():

    flag = SimpleNamespace()
    value = SimpleNamespace()
    path = SimpleNamespace()
    columns = SimpleNamespace()

    download_progress_bar_value = 0
    #data = SimpleNamespace()

    def __init__(self, config_spec: dict = {}, config_base: dict = {}):
        if(len(config_spec.keys()) == 0):
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
        
        if (len(config_base.keys()) == 0):
            self.columns.name_class = []
            self.columns.base = []
            self.columns.features = []
        
            self.path.path_sample = ""
        else:
            self.columns.name_class = config_base["name_class"]
            self.columns.base = config_base["base"]
            self.columns.features = config_base["features"]
            
            self.path.path_sample = config_base["path_sample"]
        
    


    def diff_class(self, filename: str, name_class_column: str):
        '''
        Розділяє файл ``filename`` на певну кількість файлів у залежності від значень колонки
        під назвою ``name_class_column`` які знаходяться у діапазоні від 0 до len(``name_class``)
        або мають одну зі списку ``name_class`` назву класів (задається при ініціалізації класу)
        

        Назва результуючих файлів:\n
        {``name_class``}.csv якщо присутні колонки base та features\n
        або {``name_class``}_origin.csv якщо присутні тільки колонки з назвами base\n
        .. lul
        Input:
        ------
            ``filename``            --- назва файлу\n
            ``name_class_column``   --- назва колонки в якій є маркер класу\n 
            \t\t\t\t\tу форматі від 0 до len(``name_class``)\n

        Output:
        -------
            EMPTY\n
        -------
        self:
        -----
            columns.base\n
            columns.features\n
            columns.name_class\n
            data.data_mass
        '''

        count = np.zeros(len(self.columns.name_class))
        #Pandas
        #self.data.data_mass = []
        #

        fin = open(filename,'r')
        first_line = fin.readline().strip('\n').split(",")
        base_index_array = get_col_list(first_line, self.columns.base, self.columns.features)
        name_class_column_index = first_line.index(name_class_column)

        header = ','.join([first_line[base_index_array[i]] for i in range(len(base_index_array))])+"\n"
        #Pandas
        #header_list = header.strip('\n').split(',')

        data_file = []
        for i in range(len(self.columns.name_class)):
            if(len(base_index_array) == len(self.columns.base)):
                f = open(f'{self.path.path_sample}/{self.columns.name_class[i]}_origin.csv','w')
            else:
                f = open(f'{self.path.path_sample}/{self.columns.name_class[i]}.csv','w')
            f.write(header)
            data_file.append(f)
            #Pandas
            #self.data.data_mass.append(pd.DataFrame(columns=header_list))

        for line in fin:
            n = line.strip('\n').split(',')
            base_array = [n[base_index_array[i]] for i in range(len(base_index_array))]
            #
            line_out = ','.join(base_array) + '\n'
            #
            try:
                label = int(n[name_class_column_index])
            except:
                if(n[name_class_column_index] in self.columns.name_class):
                    label = self.columns.name_class.index(n[name_class_column_index])
                else:
                    continue
                
            count[label] += 1
            #
            data_file[label].write(line_out)
            # with pandas
            # треба у майбутньому перевірити який з варіантів швидше працює
            #data_temp = pd.DataFrame(np.array(base_array), columns=header_list)
            #self.data.data_mass[label] = pd.concat([self.data.data_mass[label],data_temp],axis=1)

        for i in range(len(self.columns.name_class)):
            data_file[i].close()
        
        for i in range(len(self.columns.name_class)):
            print(self.columns.name_class[i],'\t\t',count[i])

    #cut dublicate
    def cut_cut_file(self,filein: str, fileout: str):
        """
        Видаляє дублікати з файла ``filein`` та фільтрує за допомогою функції ``out``
        
        Input:
        ------
            ``filein`` --- назва файлу вводу\n
            ``fileout`` --- назва файлу з результатом

        Output:
        -------
            ``index_in`` --- кількість оброблених записів\n
            ``index_out`` --- кількість вихідних записів\n
        -------
        self:
        -----
            columns.base\n
            columns.features\n
            value.gate\n
            value.filter\n
            flag.duplicate\n
        
        """
        fout = open(fileout,'w')
        #
        f = open(filein,'r')
        first_line = f.readline().strip('\n').split(',')
        col = get_col_list(first_line, self.columns.base, self.columns.features)
        print(filein)
        print(self.columns.features)
        print(col)
        #
        gc1 = col[0]
        gc2 = col[1]
        #
        gc21 = first_line.index("RAdeg")
        gc22 = first_line.index("DEdeg")
        #

        #kill duplicate algorithm
        def f1():
            first_flag, second_flag = 0, 0
            first_result_line, first_result_line = "", ""
            #
            index_in,index_out = 0,0
            result_index_out = 0
            #
            i=1
            z=0
            l=""
            #
            i2=1
            z2=0
            l2=""
            for line in open(filein):
                n = line.split(',')
                #duplicate algorithm
                if (i>1):
                    if (decn!=n[gc2]) or (ran!=n[gc1]):
                        ran=n[gc1]
                        decn=n[gc2]
                        if (z==1):
                            first_result_line = str(l)
                            first_flag = 1
                            #out(str(l),col,fout,self.value.gate,self.value.filter)
                            #index_out+=1
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
                #
                if (i2>1):
                    if (decn2!=n[gc22]) or (ran2!=n[gc21]):
                        ran2=n[gc21]
                        decn2=n[gc22]
                        if (z2==1):
                            second_result_line = str(l2)
                            second_flag = 1
                            #out(str(l2),col,fout,self.value.gate,self.value.filter)
                            #index_out+=1
                        l2=str(line)
                        z2=1                    
                    else:
                        z2+=1
                else:
                    ran2=n[gc21]
                    decn2=n[gc22]
                    i2=2
                    z2=1
                    l2=str(line)
                
                if((first_flag and second_flag) and (first_result_line is second_result_line)):
                    result_index_out += out(str(first_result_line),col,fout,self.value.gate,self.value.filter)
                    first_flag, second_flag = 0, 0
                    index_out+=1

                index_in+=1
            if ((z2==1 and z==1) and (l is l2)):
                result_index_out += out(str(l),col,fout,self.value.gate,self.value.filter)
                index_out+=1
            return index_in, index_out, result_index_out
        
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
                        out(str(line),col,fout,self.value.gate,self.value.filter)
                        index_out+=1
                else:
                    ran=n[gc1]
                    decn=n[gc2]
                    i=2
                    out(str(line),col,fout,self.value.gate,self.value.filter)
                    index_out+=1
                index_in+=1
            return index_in, index_out, 0
        
        if (self.flag.duplicate):
            return f2()
        else:
            return f1()

    #request to VizieR X-match
    def req(self,catalog_name: str, filename_in: str, filename_out: str):
        """
        Перетинає файл ``filename`` з каталогом під назною ``catalog_name`` за допомогою
        X-match від VizieR та виводить результат у файл ``fout`` у форматі *.csv
        
        Input:
        ------
            ``catalog_name`` --- ім'я каталогу у форматі VizieR / або ім'я файлу\n
            ``filename_in`` --- ім'я файлу вводу\n
            ``filename_out`` --- ім'я файлу виводу
        
        Output:
        -------
            EMPTY\n
        -------
        self:
        -----
            value.multi_thr\n
            value.radius
        """

        if(os.path.isfile(f'{catalog_name}.csv')):
            c = Table.read(f'{catalog_name}.csv', format = 'ascii.csv')
            c_votable = from_table(c)
            writeto(c_votable,f'{catalog_name}.vot')

        t = Table.read(f'{filename_in}.csv', format = 'ascii.csv') 
        votable = from_table(t)
        writeto(votable, f'{filename_in}.vot')
        #catalog_name = globals()[cat_name]

        WAIT_UNTIL_REPEAT_ACCESS = self.value.multi_thr["WAIT_UNTIL_REPEAT_ACCESS"]
        NUM_URL_ACCESS_ATTEMPTS = self.value.multi_thr["NUM_URL_ACCESS_ATTEMPTS"]

        attempts = 0
        while attempts < NUM_URL_ACCESS_ATTEMPTS:
            try:
                if(not os.path.isfile(f'{catalog_name}.csv')):
                    r = requests.post('http://cdsxmatch.u-strasbg.fr/xmatch/api/v1/sync',
                                    data={  'request': 'xmatch', 
                                            'distMaxArcsec': self.value.radius, 
                                            'RESPONSEFORMAT': 'csv',
                                            'cat2': f'vizier:{catalog_name}', 'colRA1': 'RA', 'colDec1': 'DEC'},
                                            files={'cat1': open(f'{filename_in}.vot', 'r')})
                    #print(r.text)
                else:
                    r = requests.post('http://cdsxmatch.u-strasbg.fr/xmatch/api/v1/sync',
                                    data={  'request': 'xmatch',
                                            'distMaxArcsec': self.value.radius, 
                                            'RESPONSEFORMAT': 'csv',
                                            'colRA1': 'RA', 'colDec1': 'DEC',
                                            'colRA2': 'RA', 'colDec2': 'DEC'},
                                            files={'cat1': open(f'{filename_in}.vot', 'r'),
                                                   'cat2': open(f'{catalog_name}.vot', 'r')})

                if(r.ok):
                    os.remove(f'{filename_in}.vot')
                    if(os.path.isfile(f'{catalog_name}.csv')):
                        os.remove(f'{catalog_name}.vot')
                    h = open(f'{filename_out}', 'w')
                    h.write(r.text)
                    h.close()
                    #print(r.text)
                    break
                else:
                    raise Exception(r.raise_for_status())
            except:
                time.sleep(WAIT_UNTIL_REPEAT_ACCESS)
                attempts += 1
        
        if(attempts == NUM_URL_ACCESS_ATTEMPTS):
            raise Exception('NUM_URL_ACCESS_ATTEMPTS')

        


    #download from VizieR
    def download(self,catalogs_name_list: list, filepath: str):
        """
        Ототожнює значення з файлу ``filepath`` з каталогами VizieR назви яких зазначені у ``catalogs_name_list`` 
        за допомогою функції ``req`` та видаляє дублікати і фільтрує об'єкти завдяки ``cut_cut_file`` 

        Input:
        ------
            ``catalogs_name_list`` --- список назв каталогів\n
            ``filepath`` --- назва фалу

        Output:
        -------
            ``count_mass`` --- список кількості значень отриманих після ототожнення (``req``) та
              фільтрації (``cut_cut_file``) для кожного з каталогів окремо\n
        -------
        self:
        -----
            value.catalogs\n
            flag.remove\n
            
            req\n
            cut_cut_file
        """
        
        filepath_temp = filepath
        count_mass = np.zeros(len(catalogs_name_list)*3)
        columns = []
        for n, name in enumerate(catalogs_name_list):
            fin = filepath_temp.split(".")[0]
            temp = f"{fin}_{name}.csv"
            self.req(self.value.catalogs["VizieR"][n],fin,temp)
            
            if(n==0 and self.flag.remove["slice"]):
                os.remove(filepath_temp)
            if(not n==0 and self.flag.remove["catalogs_cross_cut_duplicate"][n-1]):
                os.remove(filepath_temp)
            
            filepath_temp = f"{temp.split('.')[0]}_cut.csv"
            
            #
            count_mass[n*3],count_mass[n*3+1],count_mass[n*3+2] = self.cut_cut_file(temp,filepath_temp)
            
            if(self.flag.remove["catalog_cross_origin"][n]):
                os.remove(temp)

            columns.extend([name,f'{name}_cut',f'{name}_cut_filter'])

            self.download_progress_bar_value += 1

        count_mass = pd.DataFrame([np.array(count_mass)], columns=columns)

        return count_mass
        

    def multi_thr_slice_download(self,catalogs_name: list, foldername: str, filename_list: list):
        """
        Запускає процес ототожнення списку файлів ``filelist``, 
        які знаходяться у папці ``filename`` 
        з каталогами зі списку ``catalogs_name`` у багатопоточному режимі, 
        який регулюється змінною ``value.multi_thr``

        Input:
        ------
            ``catalogs_name``   --- список назв каталогів\n
            ``foldername``      --- назва папки в якій зберігаються файли з filelist\n
            ``filename_list``   --- список назв файлів
        Output:
        -------
            ``stat`` --- таблиця значень кількості рядків у файлах в залежності від етапу їх обробки\n
        -------
        self:
        -----
            download
            value.multi_thr
        """        

        MAX_WORKERS = self.value.multi_thr["MAX_WORKERS"]
        
        count_mass = []  
        try:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                for name in filename_list:
                    count_mass_temp = executor.submit(self.download,catalogs_name,f"{foldername}/{name}")
                    #print(count_mass_temp)
                    count_mass.append(count_mass_temp)
        except:
            raise Exception("Except: download part have issue. \nCheck your Ethernet conection, \nor config->flags->data_downloading variable, \nor origin catalog purity")
        
        #from threading import Thread
        #t = Thread(target=loading_progress_bar, args=[(100 / (len(catalogs_name)*len(filename_list))) * self.download_progress_bar_value])
        #t.start()

        stat = pd.DataFrame()
        for i in range(len(count_mass)):
            stat = pd.concat([stat,count_mass[i].result()],ignore_index=True)
        
        #t.s

        return stat
    
    def unslice(self,filename_list: list,foldername: str,filename_out: str):
        """
        Згортання фалів зі списку ``filename_list`` у папці ``foldername`` та 
        запис в один файл під назвою ``filename_out``

        Input:
        ------
            ``filename_list``   --- назви файлів\n
            ``foldername``      --- назва папки з файлами\n
            ``filename_out``    --- назва вихідного файлу

        Output:
        -------
            EMPTY\n
        -------
        self:
        -----
            flag.remove
        """
        
        f = open(filename_out,"w")
        for n, name in enumerate(filename_list):
            f_slice = open(f"{foldername}/{name}",'r')
            first_line = f_slice.readline()
            if(n==0):
                f.write(first_line)
            for line in f_slice:
                if (line=='\n'):
                    continue
                f.write(line)
            f_slice.close()
            
            if(self.flag.remove["catalogs_cross_cut_duplicate"][-1]):
                os.remove(f"{foldername}/{name}")
        
        if(self.flag.remove["dir"]):
            import shutil
            shutil.rmtree(foldername)
        
        f.close()

    def class_download(self,filename_in: str,filename_out: str):
        """
        Повний цикл перетину файлу під назвою ``filename`` з каталогами ``value.catalogs``
        
        Input:
        ------
            ``filename_in``  --- ім'я файлу вводу \n
            ``filename_out`` --- ім'я файлу виводу

        Output:
        -------
            ``stat_count`` --- таблиця значень кількості рядків у файлах в залежності від етапу їх обробки\n
        -------
        self:
        -----
            value.catalogs \n
            value.slice_count \n
            flag.remove \n
            path.path_sample\n
            
            multi_thr_slice_download \n
            unslice \n
        """

        #differentiation origin catalog to slice
        temp_foldername = f"{self.path.path_sample}/{filename_in.split('.')[0].split('/')[-1]}"
        stat_count, filelist = slice(filename_in,temp_foldername, self.value.slice_count)
        stat_count = pd.DataFrame(np.array(stat_count), columns=['origin'])
        #cross-match feat. VizieR
        count_cat = self.multi_thr_slice_download(self.value.catalogs["name"],temp_foldername,filelist)
        stat_count = pd.concat([stat_count,count_cat],axis=1)
        #
        filename_list = []
        for i in range(stat_count.shape[0]):
            line = str(i)
            for j in range(2,stat_count.shape[1] - 1,3):
                line += "_" + stat_count.columns.values[j]
            filename_list.append(f'{line}.csv')
        #
        print(filename_list)
        self.unslice(filename_list,temp_foldername,filename_out)
        #    
        if(self.flag.remove["origin"]):
            os.remove(filename_in)

        return stat_count

    def all_class_download(self,path_statistic: str):
        """
        Запускає процес ототожнення для всіх класів за початковим принципом

        Input:
        ------
            ``path_statistic`` --- шлях до двох файлів зі статистикою\n

        Output:
        -------
            EMPTY\n
        -------
        self:
        -----
            columns.name_class\n
            path.path_sample\n

            class_download\n
        """
        sum_mass = pd.DataFrame()

        for name in self.columns.name_class:
            if(os.path.isfile(f"{self.path.path_sample}/{name}_origin.csv")):
                stat = self.class_download(f"{self.path.path_sample}/{name}_origin.csv",f"{self.path.path_sample}/{name}.csv")
                stat.to_csv(f'{path_statistic}/{name}_slice.log', index=False)
                sum = pd.DataFrame([np.array(stat.sum(axis=0))], columns = stat.columns.values, index = [name])
                sum_mass = pd.concat([sum_mass,sum], ignore_index=False)
            else:
                raise Exception(f"File {self.path.path_sample}/{name}_origin.csv do not exist")
        
        sum_mass.to_csv(f'{path_statistic}/classes.log')

    #

        