# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 13:33:58 2020
@author: кирил
"""
import math
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

def swap(i,j):
    return j,i

def qqall(data,i):
    q_low=data[i].quantile(0.05)
    q_hi=data[i].quantile(0.95)
    rezult=data[(data[i] < q_hi) & (data[i] > q_low)]    
    return rezult
def qq(data):
    q_low=data.quantile(0.01)
    q_hi=data.quantile(0.99)
    rezult=data[(data < q_hi) & (data > q_low)]    
    return rezult
def qqm(data1,data2):
    q1_low=data1.quantile(0.01)
    q1_hi=data1.quantile(0.99)
    q2_low=data2.quantile(0.01)
    q2_hi=data2.quantile(0.99)
    '''
    rezult1,rezult2=[],[]
    for i,j in data1,data2:
        if ((i < q1_hi) and (i > q1_low)) or ((j < q2_hi) and (j > q2_low)) :
            rezult1.append(i)
            rezult2.append(j)
    #rezult=data[(data < q_hi) & (data > q_low)]    
    '''
    rezult1=data1[(data1 < q1_hi) & (data1 > q1_low) & (data2 < q2_hi) & (data2 > q2_low)]
    rezult2=data2[(data1 < q1_hi) & (data1 > q1_low) & (data2 < q2_hi) & (data2 > q2_low)]
    
    return rezult1,rezult2

#c=list(mcolors.CSS4_COLORS)
c=list(mcolors.TABLEAU_COLORS)
def Graf(ix,iy,jx,jy,q,w,e,r,name_col1,name_col2,name_col3,name_col4,save_path):  
    fig=plt.figure()
    ax = fig.add_subplot(111)
    x=q-w
    y=e-r
    ax.set_xlabel(name_col1+"-"+name_col2,fontsize=40)
    ax.set_ylabel(name_col3+"-"+name_col4,fontsize=40)
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)
    fig.set_size_inches(20,20)
    #ax.xaxis.label.set_size(20) 
    rez1,rez2=qqm(x,y)
    ax.scatter(rez1,rez2,s=20,c=c[2])
    ax.legend(loc=1,prop={'size': 20})
    fig.savefig(save_path+"/"+str(ix)+str(iy)+str(jx)+str(jy)+'.png')
    plt.close(fig)
#plt.show()

def Many_Graf_diff(data,name,save_path,count_column):
    n=count_column
    for i1 in range(n):
        for i2 in range(i1,n,1):
            for j1 in range(n):
                for j2 in range(j1,n,1):
                    if(not i1==i2 and not j1==j2 and not i1==j1 and not i2==j2 and i1+i2>j1+j2):
                        Graf(i1+1,i2+1,j1+1,j2+1,data[:][i1],data[:][i2],data[:][j1],data[:][j2],name[i1],name[i2],name[j1],name[j2],save_path)

def Many_Graf(data,name,save_path,count_column):
    n=count_column
    for i1 in range(n):
        for i2 in range(i1,n,1):
            if(not i1==i2):
                Graf(i1+1,i2+1,0,0,data[i1],0,data[i2],0,name[i1],"main",name[i2],"main",save_path)

def Graf_m(ax,q,w,e,r,index,name):
    x=q-w
    y=e-r
    rez1,rez2=qqm(x,y)
    ax.scatter(rez1,rez2,s=5,c=c[index],label=name)

def Many_Graf_many(data1,data2,data_name,name,save_path,count_column):
    n=count_column
    for i1 in range(n):
        for i2 in range(i1,n,1):
            if(not i1==i2):
                fig=plt.figure()
                ax = fig.add_subplot(111)
                ax.set_xlabel(name[i1],fontsize=40)
                ax.set_ylabel(name[i2],fontsize=40)
                fig.set_size_inches(20,20)
                Graf_m(ax,data1[name[i1]],0,data1[name[i2]],0,3,data_name[0])
                Graf_m(ax,data2[name[i1]],0,data2[name[i2]],0,2,data_name[1])
                ax.legend(loc=1,prop={'size': 20})
                fig.savefig(save_path+"/"+str(i1+1)+str(i2+1)+'.png')
                plt.close(fig)

def Many_Graf_pd(data,save_path):
    name = data['name'].unique()
    name=np.flip(name)
    name_list = data['name']
    data_ = data.drop(['name'],axis=1)
    name_column = np.array(data_.columns.values)
    print(name_column)
    n = len(name_column)
    for i1 in range(n):
        for i2 in range(i1,n,1):
            if(not i1==i2):
                fig=plt.figure()
                ax = fig.add_subplot(111)
                ax.set_xlabel(name_column[i1],fontsize=40)
                ax.set_ylabel(name_column[i2],fontsize=40)
                fig.set_size_inches(20,20)
                index=2
                #print(data_.columns.values)
                for name_ in name:
                    data_0 = data_[name_list == name_]
                    Graf_m(ax,data_0[name_column[i1]],0,data_0[name_column[i2]],0,index,name_)
                    index+=1 
                ax.legend(loc=1,prop={'size': 20})
                fig.savefig(save_path+"/"+str(i1+1)+str(i2+1)+'.png')
                plt.close(fig)

def Many_Graf_pd_diff(data,save_path):
    name = data['name'].unique()
    #name=np.flip(name)
    name_list = data['name']
    data_ = data.drop(['name'],axis=1)
    name_column = np.array(data_.columns.values)
    print(name_column)
    n = len(name_column)
    for i1 in range(n):
        for i2 in range(i1,n,1):
            for j1 in range(n):
                for j2 in range(j1,n,1):
                    if(not i1==i2 and not j1==j2 and not i1==j1 and not i2==j2 and i1+i2>j1+j2):
                        fig=plt.figure()
                        ax = fig.add_subplot(111)
                        name_axis_x = name_column[i1] + "-" + name_column[i2]
                        name_axis_y = name_column[j1] + "-" + name_column[j2]
                        ax.set_xlabel(name_axis_x,fontsize=40)
                        ax.set_ylabel(name_axis_y,fontsize=40)
                        fig.set_size_inches(20,20)
                        index=2
                        #print(data_.columns.values)
                        for name_ in name:
                            data_0 = data_[name_list == name_]
                            Graf_m(ax,data_0[name_column[i1]],data_0[name_column[i2]],data_0[name_column[j1]],data_0[name_column[j2]],index,name_)
                            index+=1
                        ax.legend(loc=1,prop={'size': 20})
                        fig.savefig(save_path+"/"+name_axis_x+"_"+name_axis_y+'.png')
                        plt.figure().clear()
                        #plt.close(fig)


def Many_Graf_diff_many(data1,data2,data_name,name,save_path,count_column):
    n=count_column
    for i1 in range(n):
        for i2 in range(i1,n,1):
            for j1 in range(n):
                for j2 in range(j1,n,1):
                    if(not i1==i2 and not j1==j2 and not i1==j1 and not i2==j2 and i1+i2>j1+j2):
                        fig=plt.figure()
                        ax = fig.add_subplot(111)
                        ax.set_xlabel(name[i1]+"-"+name[i2],fontsize=40)
                        ax.set_ylabel(name[j1]+"-"+name[j2],fontsize=40)
                        fig.set_size_inches(20,20)
                        Graf_m(ax,data1[i1],data1[i2],data1[j1],data1[j2],3,data_name[0])
                        Graf_m(ax,data2[i1],data2[i2],data2[j1],data2[j2],2,data_name[1])
                        ax.legend(loc=1,prop={'size': 20})
                        fig.savefig(save_path+"/"+str(i1+1)+str(i2+1)+str(j1+1)+str(j2+1)+'.png')
                        plt.close(fig)
            
def z_distribution(data,save_path,func):
    for i in np.arange(0,1,0.05):
        slice_d = data[(data.z > i) & (i+0.1 > data.z)]
        slice_d = slice_d.drop(['z'], axis=1)
        save_path_ = f"{save_path}/{i}"
        if not os.path.isdir(save_path_):
            os.mkdir(save_path_)
        func(slice_d,save_path_)

def z_distribution_one_img(data,save_path):
    n = len(data.columns.values) -1
    for i1 in range(n):
        for i2 in range(i1,n,1):
            for j1 in range(n):
                for j2 in range(j1,n,1):
                    if(not i1==i2 and not j1==j2 and not i1==j1 and not i2==j2 and i1+i2>j1+j2):
                        fig=plt.figure()
                        ax = fig.add_subplot(111)
                        fig.set_size_inches(20,20)
                        index=0
                        #print(data_.columns.values)
                        for i in np.arange(0,1,0.1):
                            slice_d = data[(data.z > i) & (i+0.1 > data.z)]
                            slice_d = slice_d.drop(['z'], axis=1)
                            name_column = np.array(slice_d.columns.values)
                            print(name_column)
                            Graf_m(ax,slice_d[name_column[i1]],slice_d[name_column[i2]],slice_d[name_column[j1]],slice_d[name_column[j2]],index,i)
                            index+=1
                        name_axis_x = name_column[i1] + "-" + name_column[i2]
                        name_axis_y = name_column[j1] + "-" + name_column[j2]
                        ax.set_xlabel(name_axis_x,fontsize=40)
                        ax.set_ylabel(name_axis_y,fontsize=40)
                        ax.legend(loc=1,prop={'size': 20})
                        fig.savefig(save_path+"/"+name_axis_x+"_"+name_axis_y+'.png')
                        plt.close(fig)
################
'''
for i1 in range(7):
    for i2 in range(i1,7,1):
        for j1 in range(7):
            for j2 in range(j1,7,1):
                if(not i1==i2 and not j1==j2 and not i1==j1 and not i2==j2 and i1+i2>j1+j2):
                    #q,w=swap(i1,i2)
                    Graf(i1+2,i2+2,j1+2,j2+2)
'''
#################
#Graf(11,10,11,10)
'''
for i1 in range(12):
    for i2 in range(i1,12,1):
        for j1 in range(12):
            for j2 in range(j1,12,1):
                if(not i1==i2 and not j1==j2 and not i1==j1 and not i2==j2 and i1+i2>j1+j2):
                    #q,w=swap(i1,i2)
                    Graf(i1,i2,j1,j2)
'''
'''
plt.xlabel(num_col1[0]+"-"+num_col1[1])
plt.ylabel(num_col1[1]+"-"+num_col1[2])
plt.scatter(data_nc[num_col1[0]]-data_nc[num_col1[1]],data.values()_nc[num_col1[1]]-data_nc[num_col1[2]],marker=".")
plt.savefig('E:\GRAFICS\plot.png')
#fig(num=None,figsize=(8,6),dpi=80,facecolor='w',edgecolor='k')
fig.savefig()
'''
#Graf(3,4,2,3)

#print(data[num_col])
#print(data1)
#print(data.corr())

#scatter_matrix(data, alpha=0.05, figsize=(10, 10));

def Hist1(x,save_path,name,mag):
    fig=plt.figure()
    ax = fig.add_subplot(1,1,1)
    '''
    volmin=math.trunc(np.min(x))
    volmax=math.trunc(np.max(x))
   # print(volmin,volmax)
    gis=[0]*(volmax-volmin)
   # print(gis)
    n=np.size(x)
    x_=x.isnull()
    for i in range(n):
        if (not x_[i]):     
            #print(math.trunc(x[i]-volmin),x_[i])
            gis[math.trunc(x[i]-volmin-1)]+=1
    bins=[i for i in range(volmin,volmax,1)]
    #print(bins,gis,np.size(bins),np.size(gis))
    '''
    rez=qq(x)
    fig.suptitle(name, fontsize=50)       
    ax.set_xlabel(mag,fontsize=40)
    ax.set_ylabel("count",fontsize=40)
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)
    #ax.set_title(name,fontsize = 50)
    fig.set_size_inches(30,20)
    ax.hist(rez,bins=200)
    fig.savefig(save_path+'/'+name+'.png')
    plt.close(fig)

def Hist2(n1,n2):
    fig=plt.figure()
    ax = fig.add_subplot(1,1,1)
    '''
    volmin=math.trunc(np.min(x))
    volmax=math.trunc(np.max(x))
   # print(volmin,volmax)
    gis=[0]*(volmax-volmin)
   # print(gis)
    n=np.size(x)
    x_=x.isnull()
    for i in range(n):
        if (not x_[i]):     
            #print(math.trunc(x[i]-volmin),x_[i])
            gis[math.trunc(x[i]-volmin-1)]+=1
    bins=[i for i in range(volmin,volmax,1)]
    #print(bins,gis,np.size(bins),np.size(gis))
    '''
    
    x=data_nc[n1]-data_nc[n2]
    rez=qq(x)
    fig.suptitle(n1+"-"+n2, fontsize=50)       
    ax.set_xlabel("mag",fontsize=40)
    ax.set_ylabel("count",fontsize=40)
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)
    #ax.set_title(name,fontsize = 50)
    fig.set_size_inches(30,20)
    #x.sort()
    ax.hist(rez,bins=200)
    fig.savefig('/media/kiril/j_08/AGN/excerpt/catalogue/Hist_m(7)/hist_'+n1+"_"+n2+'.png')
    plt.close(fig)


#print(data_nc['gmag'])
'''
col=[]
for i in data_nc.columns:
    if (not i=='Name_1' and not i=='z_1'):
        col.append(i)
        Hist1(data_nc[i],i)
#print(col[4])
'''
'''
for i1 in range(7):
    for i2 in range(i1,7,1):
        if(not i1==i2 ):
            Hist2(col[i1],col[i2])            
'''
'''
for i1 in range(7,12,1):
    for i2 in range(i1,12,1):
        if(not i1==i2 ):
            Hist2(col[i1],col[i2])            
'''            
def test_ML(a,b,c,model):
    n=100

    max = [a.max(), b.max(), c.max()]
    min = [a.min(), b.min(), c.min()]

    x = max[0]-min[0]
    y = max[1]-min[1]
    z = max[2]-min[2]

    dx = x/float(n)
    dy = y/float(n)
    dz = z/float(n)
    print(dx,dy,dz)
    data = []
    for ix in range(n):
        for iy in range(n):
            for iz in range(n):
                mass = [min[0]+dx*ix, min[1]+dy*iy, min[2]+dz*iz]
                data.append(mass)
        
    output_data = model.predict(np.array(data), 1024)

    data_1_0, data_1_1, data_1_2 = [], [], []    
    data_0_0, data_0_1, data_0_2 = [], [], []
    for i in range(n**3):
        if(output_data[i] > 0.5):
            data_1_0.append(data[i][0])
            data_1_1.append(data[i][1])
            data_1_2.append(data[i][2])
        else:
            data_0_0.append(data[i][0])
            data_0_1.append(data[i][1])
            data_0_2.append(data[i][2])            

    fig=plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('g',fontsize=40)
    ax.set_ylabel('bp',fontsize=40)
    fig.set_size_inches(20,20)
    ax.scatter3D(data_0_1,data_0_2,data_0_0,s=20,c='black',label="0")
    ax.scatter(data_1_1,data_1_2,data_1_0,s=20,c='red',label="1")
    ax.legend(loc=1,prop={'size': 20})
    fig.savefig('save_path.jpg')
    plt.close(fig)

def g3d(data):
    import numpy as np
    import matplotlib.pyplot as plt

    ax = plt.figure().add_subplot(projection='3d')

    # Plot a sin curve using the x and y axes.
    x = data['gmag']
    y = data['rmag']
    z = data['imag']
    

    # Plot scatterplot data (20 2D points per colour) on the x and z axes.
    #colors = ('r', 'g', 'b', 'k')

    #for c in colors:
    #    c_list.extend([c] * 20)

    ax.scatter(x, y, z, c='r', label='points')

    # Make legend, set axes limits and labels
    ax.legend()
    #ax.set_xlim(0, 1)
    #ax.set_ylim(0, 1)
    #ax.set_zlim(0, 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Customize the view angle so it's easier to see that the scatter points lie
    # on the plane y=0
    ax.view_init(elev=20., azim=-35, roll=0)

    plt.show()
