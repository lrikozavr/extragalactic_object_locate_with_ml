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

def DNN(features):
    input_array = Input(shape=(features,))
    layer_1 = Dense(features,activation='selu', kernel_initializer='he_uniform')(input_array)
    layer_2 = Dense(64, activation='elu', kernel_initializer='he_uniform')(layer_1)
    layer_3 = Dense(64, activation='tanh', kernel_initializer='he_uniform')(layer_2)
    layer_4 = Dense(64, activation='relu', kernel_initializer='he_uniform')(layer_3)
    layer_5 = Dense(64, activation='elu', kernel_initializer='he_uniform')(layer_4)
    layer_drop = Dropout(0.3)(layer_5)
    output_array = Dense(1, activation='selu', kernel_initializer='he_uniform')(layer_drop)

    model = Model(input_array, output_array)
    return model

def LogisticRegression(features):
    input_array = Input(shape=(features,))
    layer_1 = Dense(features,activation='tanh',kernel_initializer='he_uniform')(input_array)
    layer_2 = Dense(features,activation='elu',kernel_initializer='he_uniform')(layer_1)
    layer_3 = Dropout(.4)(layer_2)
    output_array = Dense(1, activation='sigmoid', kernel_initializer='he_uniform')(layer_3)

    model = Model(input_array,output_array)
    return model

def somemodel(features):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    input_array = Input(shape=(features,))

    return

def reconstruct_NN():
    return

from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm
from sklearn.model_selection import RandomizedSearchCV,ShuffleSplit
#from sklearn.utils.fixes import loguniform
def outlire(train,data_test,class_weight,sample_weight,name,config):

    label = np.ones(train.shape[0])
    '''
    clf = ExtraTreesClassifier(n_estimators=500,
                                 criterion='gini', 
                                 class_weight=class_weight,
                                 bootstrap=True,
                                 random_state=476,
                                 n_jobs=-1)
    '''
    '''
    params = {'C': loguniform(1e0, 1e3),
          'gamma': loguniform(1e-4, 1e-2)}
    
    clf = svm.SVR(gamma='scale',
                    kernel='rbf',
                    cache_size=500)
    clf_gs = RandomizedSearchCV(estimator=clf, param_distributions=params, 
                                n_iter=10, scoring='f1', n_jobs=-1, 
                                cv=ShuffleSplit(n_splits=1, test_size=0.2),   
                                refit=True, verbose=1)
    clf_gs.fit(X=train,y=label,sample_weight=sample_weight)
    '''
    early_stopping = keras.callbacks.EarlyStopping(
        monitor=config.hyperparam["model_variable"]["early_stopping"]["monitor"], 
        verbose=1,
        patience=2,
        mode=config.hyperparam["model_variable"]["early_stopping"]["mode"],
        restore_best_weights=config.hyperparam["model_variable"]["early_stopping"]["restore_best_weights"])

    clf_gs = LogisticRegression(train.shape[1])
    print("ok")
    clf_gs.compile(optimizer=config.hyperparam["optimizer"], loss="binary_crossentropy", metrics=METRICS)
    print("compile")
    clf_gs.fit(train,label,epochs=20,batch_size=1024,validation_split=0.3,callbacks=[early_stopping])
    
    data_test = pd.DataFrame(data_test)
    data_test["predict"] = clf_gs.predict(data_test)
    print(data_test)
    zero = pd.DataFrame(np.full((5,data_test.shape[1]-1),-20))
    print(zero)
    try:
        pred = clf_gs.predict(zero)
        print(pred)
    except:
        print("aboba")
    print("data predict by NN:\n",data_test["predict"])
    data_test = data_test[data_test["predict"] > config.hyperparam["model_variable"]["outlire"]["threshold"]]
    #print(data_test)
    data_test = data_test.drop(["predict"], axis=1)
    #print(data_test)
    print(data_test)
    SaveModel(clf_gs,config.path_model,config.path_weight,f'outlire_{name}')
    return np.array(data_test), data_test.index



def redshift_predict(train,label,X_test,y_test,name,config):
    from sklearn.preprocessing import normalize
    features_count = train.shape[1]

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="mean_squared_error", 
        verbose=1,
        patience=15,
        mode="min",
        restore_best_weights=True)
    
    train = normalize(train, axis=0)
    print("normalize complete")
    model_red = DNN(features_count)
    model_red.compile(optimizer='adam',
                      loss=tf.keras.losses.MeanAbsoluteError(reduction="auto", name="mean_absolute_error"),
                      metrics=tf.keras.metrics.MeanSquaredError(name="mean_squared_error", dtype=None))

    model_red.fit(train,label,
                  epochs=200,
                  batch_size=1024,
                  validation_split=0.3,
                  callbacks=[early_stopping])
    print("model_redshift fited")

    predict_red = model_red.predict(X_test, batch_size=1024)

    predict_red = pd.DataFrame(np.array(predict_red), columns=['redshift_pred'])
    y_test = pd.DataFrame(np.array(y_test),columns=['actual_redshift'])
    predict_red = pd.concat([predict_red,y_test], axis=1)
    
    predict_red.to_csv(f"{config.path_predict}_{name}_redshift.csv")

    return model_red

def model_volume(train,label,X_train,y_train,X_test,y_test,
	model,optimizer,loss,sample_weight,class_weights,num_ep,batch_size,validation_split,
    path_save_eval,name,config):

    early_stopping = keras.callbacks.EarlyStopping(
        monitor=config.hyperparam["model_variable"]["early_stopping"]["monitor"], 
        verbose=1,
        patience=config.hyperparam["model_variable"]["early_stopping"]["patience"],
        mode=config.hyperparam["model_variable"]["early_stopping"]["mode"],
        restore_best_weights=config.hyperparam["model_variable"]["early_stopping"]["restore_best_weights"])

    model.compile(optimizer=optimizer, loss=loss, metrics=METRICS, weighted_metrics=['accuracy']) #!
    print("model compiled")
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
        callbacks=[early_stopping],
        class_weight=class_weights,
        #&&&????????????????????????????????????????????????????????????????????
        sample_weight=sample_weight
        )
    
    print("model fited")
    #model.evaluate(X_test, y_test, verbose=1)
    #model.summary()
    SaveModel(model,f'{config.path_model}',f'{config.path_weight}',name)
    model = LoadModel(f'{config.path_model}_{name}',f'{config.path_weight}_{name}',optimizer,loss)

    if(config.hyperparam["model_variable"]["metric_culc"] == "test"):
        if(config.hyperparam["model_variable"]["outlire"]["work"]):
            data_test, outlire_index = outlire(X_train,X_test,None,sample_weight,name,config)
            pd_label = pd.DataFrame(np.array(y_test[outlire_index]), columns=config.name_class_cls)
        else:
            data_test = X_test
            pd_label = pd.DataFrame(np.array(y_test), columns=config.name_class_cls)
        Class = model.predict(data_test, batch_size)
    else:
        Class = model.predict(train, batch_size)
        pd_label = pd.DataFrame(np.array(label), columns=config.name_class_cls)
    #param from config (columns)
    res = pd.DataFrame(np.array(Class), columns=config.name_class_prob)
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

def NN(train,label,red_label,sample_weight,validation_split,batch_size,num_ep,optimizer,loss,class_weights,
output_path_predict,output_path_mod,output_path_weight,path_save_eval,config):
    
    features = train.shape[1]
    initial_weights = os.path.join(tempfile.mkdtemp(), 'initial_weights')

    def cust_model(name,X_train,y_train,X_test,y_test):
        n = name.split("n")
        l2,l3,l4 = n[1],n[2],n[3]
        a = config.hyperparam["model_variable"]["activation"]
        model = DeepCustomNN_sm(features,l2,l3,l4,a[0],a[1],a[2],len(config.name_class))	
        
        model1 = model_volume(train,label,X_train,y_train,X_test,y_test,
        model,optimizer,loss,sample_weight,class_weights,num_ep,batch_size,validation_split,
        path_save_eval,f"custom_sm_{name}",config)
        
        #model_red = redshift_predict(X_train,red_train,X_test,red_test,name,config=config)
        #SaveModel(model_red,f'{config.path_model}_redshift',f'{config.path_weight}_redshift',name)

        #SaveModel(model1,output_path_mod,output_path_weight,f"custom_sm_{name}")
    
    def model_activate(flag,custom_index,X_train,y_train,X_test,y_test):
        #param from config 'gpu', 'cpu'
        print(flag)
        if(flag == 'gpu'):
            for name in custom_index:
                cust_model(name,X_train,y_train,X_test,y_test)
        #add multithread resolve
        if(flag == 'cpu'):
            import multiprocessing
            MAX_WORKERS = multiprocessing.cpu_count()
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                for name in custom_index:
                    executor.submit(cust_model,name,X_train,y_train,X_test,y_test)
    
    #hyperparam from config
    kfold = KFold(n_splits=config.hyperparam["model_variable"]["kfold"], shuffle=False)
    index=0
    for train_index, test_index in kfold.split(train):
        #print("train len ",len(train_index),"test len ",len(test_index))
        #train len  358845 test len  89712
        X_train = train[train_index]
        y_train = label[train_index]
        red_train = red_label[train_index]
    
        X_test = train[test_index]
        y_test = label[test_index]
        red_test = red_label[test_index]
        
        custom_index = []
        #print(index)
        #hyperparam from config
        custom_index.append(make_custom_index(index,config.hyperparam["model_variable"]["neuron_count"]))
        #print(custom_index)
        model_activate(config.flags["system"],custom_index,X_train,y_train,X_test,y_test)
        index+=1
    #
    index = '00'
    custom_index = []
    #hyperparam from config
    custom_index.append(make_custom_index(index,config.hyperparam["model_variable"]["neuron_count"]))
    model_activate(config.flags["system"],custom_index,train,label,train,label)


def large_file_prediction(config):
    from data_process import get_features, deredded

    def DataTransform(data,config):
        data_new = pd.DataFrame()
        for features_flag in config.features["train"]:
            match features_flag:
                #може потрібно тут ставити запобіжник?
                case "color":
                    mags = get_features(["mags"],config)
                    num_colours = sum(i for i in range(len(mags)))
                    colours = np.zeros((data.shape[0],num_colours))
                    data_temp = np.array(data[mags].values)
                    #print("data_temp",data_temp)
                    index=0
                    for j in range(len(mags)):
                        for i in range(j, len(mags)):
                            if(i!=j):
                                colours[:,index] = data_temp[:,j] - data_temp[:,i]
                                index+=1
                    #print(colours)
                    color = get_features(["color"],config)
                    colours = pd.DataFrame(colours,columns=color)
                    data_new = pd.concat([data_new,colours],axis=1)
                    del data_temp
                    del colours
                case "mags":
                    mags = get_features(["mags"],config)
                    data_new = pd.concat([data_new,data[mags]],axis=1)                    
                case "err_color":
                    mags = get_features(["err_mags"],config)
                    num_colours = sum(i for i in range(len(mags)))
                    colours = np.zeros((data.shape[0],num_colours))
                    data_temp = np.array(data[mags])
                    index=0
                    for j in range(len(mags)):
                        for i in range(j, len(mags)):
                            if(i!=j):
                                colours[:,index] = data_temp[:,j] - data_temp[:,i]
                                index+=1
                    err_color = get_features(["err_color"],config)
                    colours = pd.DataFrame(colours,columns=err_color)
                    data_new = pd.concat([data_new,colours],axis=1)
                    del data_temp
                    del colours
                case "err_mags":
                    mags_err = get_features(["err_mags"],config)
                    data_new = pd.concat([data_new,data[mags_err]],axis=1)                    
                case _:
                    raise Exception('unknown config value config.features["train"]')
        
        del data
        
        #print(data_new.columns.values)
        #print(data_new)
        #data_new.to_csv('/home/lrikozavr/ML_work/test/ml/prediction/data_new.csv')
        if(config.flags['data_preprocessing']['main_sample']['normalize']['work']):
            norms = pd.read_csv(f"{config.path_stat}/{config.name_main_sample}_norms.csv")
            data_new[norms.columns.values] = data_new[norms.columns.values].div(np.array(norms))

        if(config.flags["prediction"]["outlire"]):
            outlire_model = LoadModel(f'{config.path_model}_outlire_{name}',f'{config.path_weight}_outlire_{name}',config.hyperparam["optimizer"],"binary_crossentropy")
            data_new['outlire_prob'] = outlire_model.predict(data_new)
            data_new = data_new[data_new['outlire_prob'] > config.hyperparam["model_variable"]["outlire"]["threshold"]].drop(['outlire_prob'], axis=1)

        return data_new
        #get_features(config)

    def ml(output_path_mod,output_path_weight,data,config):
        model = LoadModel(output_path_mod,output_path_weight,config.hyperparam["optimizer"],config.hyperparam["loss"])
        print(data)
        data_temp = deredded(data.replace('null',0.0),config)
        print("deredded")
        data_temp_tr = data_temp[config.features['data']].astype(float)
        print("cut null")
        del data_temp
        data_transform = DataTransform(data_temp_tr,config)
        print("Data Transform")
        del data_temp_tr
        predicted = model.predict(data_transform, config.hyperparam["batch_size"])
        print("predicted")
        del data_transform
        predicted = pd.DataFrame(np.array(predicted), columns=config.name_class_prob)
        data = pd.concat([data,predicted], axis=1)
        del predicted
        return data
    
    count = config.flags["prediction"]["batch_count"]
    i, index = 0, 1

    data_mass = [[]]*count
    #data_mass = np.array((count,))
    f = open(config.prediction_path,'r')
    columns = f.readline().strip('\n').split(",")
    
    #print(columns)
    columns_temp = ['']*len(columns)
    for i, col in enumerate(columns):
        if (col in config.base) and (i >= len(config.base)):
            columns_temp[i] = str(col+"_a")
        else:
            columns_temp[i] = str(col)
    
    columns = list(columns_temp)
    #
    for fc in config.features["data"]:
        if(not fc in columns):
            raise Exception("prediction catalog don't have features columns")
    #
    for bc in range(2):
        if(not config.base[bc] in columns):
            raise Exception("prediction catalog don't have coordinate columns")
    #
    import time
    i=0
    for line in f:
        if(i // count == index):
            index += 1
            #magic
            print(index-1,"start")
            data_mass_temp = pd.DataFrame(data_mass, columns=columns)
            name = make_custom_index('00',config.hyperparam["model_variable"]["neuron_count"])
            data = ml(f"{config.path_model}_custom_sm_{name}",f"{config.path_weight}_custom_sm_{name}",data_mass_temp,config)
            data.to_csv(f"{config.path_predict}_{name}_{index-1}.csv", index=False)
            print(index-1,"done")
            del data_mass_temp
            del data
            time.sleep(3)

        #print(i - (index-1)*count)
        line_list = list(line.strip('\n').split(","))
        #
        flag_null = 0
        for j in range(len(config.features["data"]) // 2):
            if (line_list[columns.index(config.features["data"][j*2])] == 'null'):
                print(i," --- column have null features value")
                flag_null = 1
                break
        if(flag_null):
            continue
        #
        try:
            float(line_list[columns.index(config.base[0])])
            float(line_list[columns.index(config.base[1])])
        except ValueError:
            print(i," --- line contain bad 'ra', 'dec' value")
            continue
        if(len(line_list) == len(columns)):
            data_mass[i - (index-1)*count] = line_list
            i += 1
    
    #print(data_mass[200][1])
    data_mass = pd.DataFrame(data_mass, columns=columns)
    data_mass = pd.DataFrame(data_mass.head(i - (index-1)*count))
    name = make_custom_index('00',config.hyperparam["model_variable"]["neuron_count"])
    data = ml(f"{config.path_model}_custom_sm_{name}",f"{config.path_weight}_custom_sm_{name}",data_mass,config)
    data.head(i - (index-1)*count).to_csv(f"{config.path_predict}_{name}_{index}.csv", index=False)
    del data