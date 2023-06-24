# -*- coding: utf-8 -*-

import pandas as pd
import os
import tempfile

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

#from main import hyperparam, flags, name_class

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


def DeepCustomNN_sm(features, l2, l3, l4, a2, a3, a4, output): #16, 8, 4
    input_array = Input(shape=(features,))
    layer_1 = Dense(features, activation='linear', kernel_initializer='he_uniform')(input_array)
    layer_2 = Dense(l2, activation=a2, kernel_initializer='he_uniform' )(layer_1)
    layer_3 = Dense(l3, activation=a3, kernel_initializer='he_uniform' )(layer_2)
    layer_4 = Dense(l4, activation=a4, kernel_initializer='he_uniform' )(layer_3)
    #layer_last = Dropout(.2)(layer_4)
    output_array = Dense(output, activation='softmax', kernel_initializer='he_uniform')(layer_4)

    #Выдает прописной вариант названия активатора
    #keras.activations.serialize(keras.activations.hard_sigmoid)

    model = Model(input_array, output_array)
    return model

def somemodel(features):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    input_array = Input(shape=(features,))

    return

def reconstruct_NN():
    return

def model_volume(train,label,X_train,y_train,X_test,y_test,
	model,optimizer,loss,class_weights,num_ep,batch_size,validation_split,
	output_path_predict,path_save_eval,name,config):
	
    model.compile(optimizer=optimizer, loss=loss, metrics=METRICS) #!
    #model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

    '''
    #load weights values from begin
    initial_weights = os.path.join(tempfile.mkdtemp(), 'initial_weights')
    model.save_weights(initial_weights)
    model.load_weights(initial_weights)
    '''
    model.fit(X_train, y_train,
        epochs=num_ep,
        #verbose=1,
        verbose=0,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stopping]
        #class_weight=class_weights
        #&&&????????????????????????????????????????????????????????????????????
        #sample_weight=
        )
    #model.evaluate(X_test, y_test, verbose=1)
    #model.summary()



    if(config.hyperparam["model_variable"]["metric_culc"] == "test"):
        Class = model.predict(X_test, batch_size)
        pd_label = pd.DataFrame(np.array(y_test), columns=config.name_class_cls)
    else:
        Class = model.predict(train, batch_size)
        pd_label = pd.DataFrame(np.array(label), columns=config.name_class_cls)
    #param from config (columns)
    res = pd.DataFrame(np.array(Class), columns=config.name_class_cls_prob)
    #print(pd_label)
    res = pd.concat([res, pd_label], axis=1)
    
    res.to_csv(f'{path_save_eval}_{name}_prob.csv', index=False)
    
    return model

def make_custom_index(index,list):
    name = f'{index}n'
    for i in range(len(list)):
        if(i+1 == len(list)):
            name += str(list[i])
        else:
            name += str(list[i]) + 'n'
    return name

def NN(train,label,validation_split,batch_size,num_ep,optimizer,loss,class_weights,
output_path_predict,output_path_mod,output_path_weight,path_save_eval,config):
    
    features = train.shape[1]
    initial_weights = os.path.join(tempfile.mkdtemp(), 'initial_weights')

    def cust_model(name,X_train,y_train,X_test,y_test):
        n = name.split("n")
        l2,l3,l4 = n[1],n[2],n[3]
        a = config.hyperparam["model_variable"]["activation"]
        model = DeepCustomNN_sm(features,l2,l3,l4,a[0],a[1],a[2],3)	
        
        model1 = model_volume(train,label,X_train,y_train,X_test,y_test,
        model,optimizer,loss,class_weights,num_ep,batch_size,validation_split,
        output_path_predict,path_save_eval,f"custom_sm_{name}",config)
        SaveModel(model1,output_path_mod,output_path_weight,f"custom_sm_{name}")
    
    def model_activate(flag,custom_index,X_train,y_train,X_test,y_test):
        #param from config 'gpu', 'cpu'
        if('gpu'):
            for name in custom_index:
                cust_model(name,X_train,y_train,X_test,y_test)
        #add multithread resolve
        if('cpu'):
            import multiprocessing
            MAX_WORKERS = multiprocessing.cpu_count()
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                for name in custom_index:
                    executor.submit(cust_model,name,X_train,y_train,X_test,y_test)
    


    #print(features)
    #print(label)
    #train = np.array(train)

    #hyperparam from config
    kfold = KFold(n_splits=config.hyperparam["model_variable"]["kfold"], shuffle=False)
    index=0
    for train_index, test_index in kfold.split(train):
        #print("train len ",len(train_index),"test len ",len(test_index))
        #train len  358845 test len  89712
        X_train = train[train_index]
        y_train = label[train_index]
    
        X_test = train[test_index]
        y_test = label[test_index]
        
        custom_index = []
        
        #hyperparam from config
        custom_index.append(make_custom_index(index,config.hyperparam["model_variable"]["neuron_count"]))
        model_activate(config.flags["system"],custom_index,X_train,y_train,X_test,y_test)
    index+=1
    #
    index = '00'
    custom_index = []
    #hyperparam from config
    custom_index.append(make_custom_index(index,config.hyperparam["model_variable"]["neuron_count"]))
    model_activate(config.flags["system"],custom_index,train,label,train,label)
