#!/home/kiril/python_env_iron_ment/new_proj/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

#Разница между всеми
def Diff(data,flag_color):
	stars = data.shape[0]
	mags = data.shape[1]
	num_colours = sum(i for i in range(mags))
	colours = np.zeros((stars,num_colours))
	index = 0
	#
	for j in range(mags):
		for i in range(j, mags):
			if(i!=j):
				colours[:,index] = data[:,j] - data[:,i]
				index += 1
	if(not flag_color):
		print("data && colors")
		#Result = np.append(data,colours, axis=1)
		Result = np.append(colours,data, axis=1)
	else:
		print("colors")
		Result = colours
	print("Different all Data")
	print(Result.shape)
	return Result

#Выравнивание
def Rou(data):
	features = data.shape[1]
	#print(features)
	#print(data)
	info = pd.DataFrame()
	means = np.zeros(features)
	stds = np.zeros(features)
	Result = np.array(data)
	for i in range(features):
		means[i] = np.mean(data[:,i])
		stds[i] = np.std(data[:,i])
		info[str(i)] = [means[i],stds[i],np.min(data[:,i]),np.max(data[:,i])]
		Result[:,i] = (data[:,i] - means[i])/stds[i]
	print("Normalisation Data")
	return Result, info
	
def DataP(data,flag_color):
	data.fillna(0)
	data.info()
	#data.sum()
	data = np.array(data)
	#return Rou(Diff(data,flag_color)) #return  Diff(data,flag_color) #return data 
	return  Diff(data,flag_color)

def z_round(data1,data2):
	data_1,data_2 = pd.DataFrame(),pd.DataFrame()
	for i in np.arange(0,1,0.1):
		slice_d_1 = data1[(data1.z > i) & (i+0.1 > data1.z)]
		slice_d_2 = data2[(data2.z > i) & (i+0.1 > data2.z)]
		
		if(len(slice_d_2) > len(slice_d_1) and (not len(slice_d_2) == 0 and not len(slice_d_1) == 0) ):
			size=len(slice_d_1)
			slice_d_2 = slice_d_2.sample(size)	
		else:
			size=len(slice_d_2)
			slice_d_1 = slice_d_1.sample(size)
		
		data_1 = data_1.append(slice_d_1, ignore_index=True)
		data_2 = data_2.append(slice_d_2, ignore_index=True)
	return data_1,data_2
			

