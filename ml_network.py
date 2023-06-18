# -*- coding: utf-8 -*-

import pandas as pd
import os
import math

from keras.layers import Input, Dense, Dropout
from keras.models import Model
from tensorflow import keras

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,KFold
import tensorflow as tf
from keras import backend as K
import numpy as np

import sklearn.metrics as skmetrics

METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_prc', 
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)



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

    #Выдает прописной вариант названия активатора
    #keras.activations.serialize(keras.activations.hard_sigmoid)

    model = Model(input_array, output_array)
    return model

def DeepCustomNN_sm(features, l2, l3, l4, output): #16, 8, 4
    input_array = Input(shape=(features,))
    layer_1 = Dense(features, activation='linear', kernel_initializer='he_uniform')(input_array)
    layer_2 = Dense(l2, activation="softsign", kernel_initializer='he_uniform' )(layer_1)
    layer_3 = Dense(l3, activation="tanh", kernel_initializer='he_uniform' )(layer_2)
    layer_4 = Dense(l4, activation="elu", kernel_initializer='he_uniform')(layer_3)
    #layer_last = Dropout(.2)(layer_4)
    output_array = Dense(output, activation='softmax', kernel_initializer='he_uniform')(layer_4)

    #Выдает прописной вариант названия активатора
    #keras.activations.serialize(keras.activations.hard_sigmoid)

    model = Model(input_array, output_array)
    return model


def DeepCustomZNN(features, l2, l3, l4):
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

def eval(y,y_pred,n,name):
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
    columns=['Accuracy',f'{name}_purity',f'non{name}_precision',f'{name}_completness',f'non{name}_completness','F1',
    'FPR','TNR','bACC','K','MCC','BinaryBS'])

	print("Accuracy 				[worst: 0; best: 1]:",              Acc)
	print(f"{name} purity 				[worst: 0; best: 1]:",     pur_a)
	print(f"non{name} precision 			[worst: 0; best: 1]:",    pur_not_a)
	print(f"{name} completness 			[worst: 0; best: 1]:",       com_a)
	print(f"non{name} completness 			[worst: 0; best: 1]:",     com_not_a)
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

def before_ev(path_save_eval,filename):
	name_col = ['star_cls','gal_cls','qso_cls']
	data = pd.read_csv(f'{path_save_eval}_{filename}_prob.csv', header=0, sep=',')
	for i in range(data.shape[0]):
		max = 0
		for name in name_col:
			if(data[f'{name}_prob'].iloc[i] > max):
				max = data[f'{name}_prob'].iloc[i]
		for name in name_col:
			if(not data[f'{name}_prob'].iloc[i] == max):
				data[f'{name}_prob'].iloc[i] = 0
			else:
				data[f'{name}_prob'].iloc[i] = 1
		#print(data.iloc[i])

	for name in name_col:
		ev = eval(np.array(data[name]),np.array(data[f'{name}_prob']),data.shape[0],name)
		ev.to_csv(f'{path_save_eval}_{filename}_{name}_evaluate.csv', index=False)



def ml_volume(train,label,X_train,y_train,X_test,y_test,
	model,optimizer,loss,class_weights,num_ep,batch_size,validation_split,
	output_path_predict,path_save_eval,name):
	
	model.compile(optimizer=optimizer, loss=loss, metrics=METRICS)
	#model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

	model.fit(X_train, y_train,
		epochs=num_ep,
		#verbose=1,
		verbose=0,
		batch_size=batch_size,
		validation_split=validation_split
		#class_weight=class_weights
		#&&&????????????????????????????????????????????????????????????????????
        #sample_weight=
		)
	model.evaluate(X_test, y_test, verbose=1)
	model.summary()

	
	Class = model.predict(X_test, batch_size)
	#Class = model.predict(train, batch_size)
	#print(Class)
	res = pd.DataFrame(np.array(Class), columns=["star_cls_prob","qso_cls_prob","gal_cls_prob"])
	#res['Y'] = np.array(label)
	#pd_label = pd.DataFrame(np.array(label), columns=["star_cls","qso_cls","gal_cls"])
	pd_label = pd.DataFrame(np.array(y_test), columns=["star_cls","qso_cls","gal_cls"])
	print(pd_label)
	res = pd.concat([res, pd_label], axis=1)
 # type: ignore	print(res)
	
	res.to_csv(f'{path_save_eval}_{name}_prob.csv', index=False)
	
	#ev = eval(Class,label,label.shape[0])
	#ev.to_csv(f'{path_save_eval}_{name}_evaluate.csv', index=False)

	return model

def NN(train,label,test_size,validation_split,batch_size,num_ep,optimizer,loss,class_weights,
output_path_predict,output_path_mod,output_path_weight,path_save_eval):

	#gpu_devices = tf.config.experimental.list_physical_devices('GPU')
	#for device in gpu_devices:
	#	tf.config.experimental.set_memory_growth(device, True)
	#from tensorflow.python.client import device_lib
	#print(device_lib.list_local_devices())
	#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
	
	features = train.shape[1]
	print(features)
	#print(label)
	train = np.array(train)
	kfold = KFold(n_splits=5, shuffle=False)
	index=0
	for train_index, test_index in kfold.split(train):
		#print("train len ",len(train_index),"test len ",len(test_index))
		#train len  358845 test len  89712
		X_train = train[train_index]
		y_train = label[train_index]
    
		X_test = train[test_index]
		y_test = label[test_index]
		
		#add multithread resolve
		custom_index = []
		'''
		#for l2 in range(12,16,1):
		for l2 in range(3200,3600,1):
			#for l3 in range(8,12,1):
			for l3 in range(10200,10600,1):
				#for l4 in range(4,8,1):
				for l4 in range(3200,3800,1):
					custom_index.append(str(index) + "n" + str(l2) + "n" + str(l3) + "n" + str(l4))
		'''
		for i in range(11):
			custom_index.append(str(index) + "n" + str(int(2**i)) + "n" + str(int(2**i)) + "n" + str(int(2**i)))

		#custom_index.append(str(index) + "n" + str(64) + "n" + str(64) + "n" + str(64))

		'''
		for l2 in range(12,17,1):
			line = str(index) + "n" + str(l2) + "n" + str(8) + "n" + str(4)
			custom_index.append(line)
		'''	
		for name in custom_index:
			n = name.split("n")
			l2,l3,l4 = n[1],n[2],n[3]
			model = DeepCustomNN_sm(features,l2,l3,l4,3)	
			#model = DeepCustomZNN(features,l2,l3,l4)	
			model1 = ml_volume(train,label,X_train,y_train,X_test,y_test,
			model,optimizer,loss,class_weights,num_ep,batch_size,validation_split,
			output_path_predict,path_save_eval,f"custom_sm_{name}")
			
			SaveModel(model1,output_path_mod,output_path_weight,f"custom_sm_{name}")
		'''
		def cust_multi(name):
			n = name.split("n")
			l2,l3,l4 = n[1],n[2],n[3]
			model = DeepCustomNN(features,l2,l3,l4)	
			model1 = ml_volume(train,label,X_train,y_train,X_test,y_test,
			model,optimizer,loss,num_ep,batch_size,validation_split,
			output_path_predict,path_save_eval,f"custom_{name}")
			SaveModel(model1,output_path_mod,output_path_weight,f"custom_{name}")

		MAX_WORKERS = 16
		from concurrent.futures import ThreadPoolExecutor
		attempts = 0
		with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
			for name in custom_index:
				executor.submit(cust_multi,name)
		'''
		'''
		model = DeepLinearNN(features)
		model2 = ml_volume(train,label,X_train,y_train,X_test,y_test,
		model,optimizer,loss,num_ep,batch_size,validation_split,
		output_path_predict,path_save_eval,f"linear_{index}")
		SaveModel(model2,output_path_mod,output_path_weight,f"linear_{index}")
		'''
		index+=1
