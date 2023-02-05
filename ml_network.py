#!/home/kiril/python_env_iron_ment/new_proj/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import os
import math

from keras.layers import Input, Dense, Dropout
from keras.models import Model
import keras

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,KFold
import tensorflow as tf
from keras import backend as K
import numpy as np

import sklearn.metrics as skmetrics

def SaveModel(model, path_model, path_weights, name):
    model_json = model.to_json()
    with open(f"{path_model}_{name}", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(f"{path_weights}_{name}")
    print("Model is saved to disk\n")

def LoadModel(path_model, path_weights, optimizer, loss):
    from keras.models import model_from_json
    json_file = open(path_model, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(path_weights)
    print("Model is loaded from disk\n")
    loaded_model.compile(optimizer=optimizer, loss=loss)
    return loaded_model

def DeepCustomNN(features, l2, l3, l4): #16, 8, 4
    input_array = Input(shape=(features,))
    layer_1 = Dense(features, activation='linear', kernel_initializer='he_uniform')(input_array)
    layer_2 = Dense(l2, activation="softsign", kernel_initializer='he_uniform' )(layer_1)
    layer_3 = Dense(l3, activation="tanh", kernel_initializer='he_uniform' )(layer_2)
    layer_4 = Dense(l4, activation="elu", kernel_initializer='he_uniform')(layer_3)
    #layer_last = Dropout(.2)(layer_4)
    output_array = Dense(1, activation='sigmoid', kernel_initializer='he_uniform')(layer_4)
    #output_array = Dense(1, activation='swish', kernel_initializer='he_uniform')(layer_last)

    #Выдает прописной вариант названия активатора
    #keras.activations.serialize(keras.activations.hard_sigmoid)

    model = Model(input_array, output_array)
    return model

def DeepCustomZNN(features):
    input_array = Input(shape=(features,))
    layer_1 = Dense(features, activation='linear', kernel_initializer='he_uniform')(input_array)
    layer_2 = Dense(16, activation="softsign", kernel_initializer='he_uniform' )(layer_1)
    layer_3 = Dense(8, activation="tanh", kernel_initializer='he_uniform' )(layer_2)
    layer_last = Dropout(.2)(layer_3)
    output_array = Dense(1, activation='swish', kernel_initializer='he_uniform')(layer_last)

    #Выдает прописной вариант названия активатора
    #keras.activations.serialize(keras.activations.hard_sigmoid)

    model = Model(input_array, output_array)
    return model

def DeepLinearNN(features):
    input_img = Input(shape=(features,))
    layer_1 = Dense(64, activation='linear', kernel_initializer='he_uniform' )(input_img)
    layer_2 = Dense(32, activation='linear', kernel_initializer='he_uniform' )(layer_1)
    layer_3 = Dense(16, activation='linear', kernel_initializer='he_uniform' )(layer_2)
    layer_4 = Dense(8, activation='linear', kernel_initializer='he_uniform' )(layer_3)
    layer_5 = Dense(4, activation='linear', kernel_initializer='he_uniform' )(layer_4)
    Label = Dense(1,activation='sigmoid', kernel_initializer='he_uniform' )(layer_5)
    model = Model(input_img, Label)
    return model

def eval(y,y_pred,n):
	count = 0
	TP, FP, TN, FN = 0,0,0,0
	for i in range(n):
		if(y[i]<0.5):
			y[i] = 0
		if(y[i]>=0.5):
			y[i] = 1
		if(y[i]==y_pred[i]):
			count+=1
		if(y[i]==1):
			if(y[i]==y_pred[i]):
				TP += 1
			else:
				FP += 1
		if(y[i]==0):
			if(y[i]==y_pred[i]):
				TN += 1
			else:
				FN += 1
	Acc = count/n
	pur_a = TP/(TP+FP)
	pur_not_a = TN/(TN+FN)
	com_a = TP/(TP+FN)
	com_not_a = TN/(TN+FP)
	f1 = 2*TP/(2*TP+FP+FN)
	fpr = FP/(TN+FN)
	tnr = TN/(TN+FN)
	bAcc = (TP/(TP+FP)+TN/(TN+FN))/2.
	k = 2*(TP*TN-FN*FP)/((TP+FP)*(FP+TN)+(TP+FN)*(FN+TN))
	mcc = (TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
	BinBs = (FP+FN)/(TP+FP+FN+TN)

	print(np.array([Acc,pur_a,pur_not_a,com_a,com_not_a,f1,fpr,tnr,bAcc,k,mcc,BinBs]))
	ev = pd.DataFrame([np.array([Acc,pur_a,pur_not_a,com_a,com_not_a,f1,fpr,tnr,bAcc,k,mcc,BinBs])], 
    columns=['Accuracy','AGN_purity','nonAGN_precision','AGN_completness','nonAGN_completness','F1',
    'FPR','TNR','bACC','K','MCC','BinaryBS'])

	print("Accuracy 				[worst: 0; best: 1]:",              Acc)
	print("AGN purity 				[worst: 0; best: 1]:",     pur_a)
	print("nonAGN precision 			[worst: 0; best: 1]:",    pur_not_a)
	print("AGN completness 			[worst: 0; best: 1]:",       com_a)
	print("nonAGN completness 			[worst: 0; best: 1]:",     com_not_a)
	print("F1  					[worst: 0; best: 1]:",		f1)
	#print("AGN_F:",     2*(Tq/(Tq+Fq)*Tq/(Tq+Fs))/(Tq/(Tq+Fq)+Tq/(Tq+Fs)) )
	#print("non_AGN_F:",     2*(Ts/(Ts+Fs)*Ts/(Ts+Fq))/(Ts/(Ts+Fq)+Ts/(Ts+Fs)) )
	print("FPR (false positive rate) 		[worst: 1; best: 0]:",		fpr)
	print("TNR (true negative rate) 		[worst: 0; best: 1]:",		tnr)
	print("bACC (balanced accuracy) 		[worst: 0; best: 1]:", bAcc)
	print("K (Cohen's Kappa) 			[worst:-1; best:+1]:",		k)
	print("MCC (Matthews Correlation Coef) 	[worst:-1; best:+1]:",		mcc)
	print("BinaryBS (Brierscore) 			[worst: 1; best: 0]:", BinBs)

	return ev

def ml_volume(train,label,X_train,y_train,X_test,y_test,
	model,optimizer,loss,num_ep,batch_size,validation_split,
	output_path_predict,path_save_eval,name):
	
	model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])
	model.fit(X_train, y_train,
		epochs=num_ep,
		verbose=1,
		batch_size=batch_size,
		validation_split=validation_split
		#&&&????????????????????????????????????????????????????????????????????
        #sample_weight=
		)
	model.evaluate(X_test, y_test, verbose=1)
	model.summary()

	

	Class = model.predict(train, batch_size)
	#print(Class)
	res = pd.DataFrame(np.array(Class), columns=['y_prob'])
	res['Y'] = np.array(label)
	res.to_csv(f'{path_save_eval}_{name}_prob.csv', index=False)

	ev = eval(Class,label,label.shape[0])
	ev.to_csv(f'{path_save_eval}_{name}_evaluate.csv', index=False)

	return model

def NN(train,label,test_size,validation_split,batch_size,num_ep,optimizer,loss,
output_path_predict,output_path_mod,output_path_weight,path_save_eval):

	features = train.shape[1]
	print(features)
	train = np.array(train)
	kfold = KFold(n_splits=5, shuffle=False)
	index=0
	for train_index, test_index in kfold.split(train):
		X_train = train[train_index]
		y_train = label[train_index]
    
		X_test = train[test_index]
		y_test = label[test_index]
		'''
		
		for l2 in range(4,16,1):
			for l3 in range(2,16,1):
				for l4 in range(1,16,1):
					custom_index = float(str(index) + "0" + str(l2) + "0" + str(l3) + "0" + str(l4))
					model = DeepCustomNN(features,l2,l3,l4)	
					model1 = ml_volume(train,label,X_train,y_train,X_test,y_test,
					model,optimizer,loss,num_ep,batch_size,validation_split,
					output_path_predict,path_save_eval,f"custom_{custom_index}")
					SaveModel(model1,output_path_mod,output_path_weight,f"custom_{custom_index}")
		'''
		model = DeepLinearNN(features)
		model2 = ml_volume(train,label,X_train,y_train,X_test,y_test,
		model,optimizer,loss,num_ep,batch_size,validation_split,
		output_path_predict,path_save_eval,f"linear_{index}")
		SaveModel(model2,output_path_mod,output_path_weight,f"linear_{index}")

		index+=1
