# -*- coding: utf-8 -*-

import pandas as pd
import os
import tempfile

import math

from keras.layers import Input, Dense, Dropout, Flatten
from keras.models import Model
from tensorflow import keras

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,KFold
import tensorflow as tf
from keras import backend as K
import numpy as np

import sklearn.metrics as skmetrics
from data_process import get_features, deredded

tf.random.set_seed(777)
keras.utils.set_random_seed(777)
np.random.seed(777)

#from main import hyperparam, flags, name_class

def loading_progress_bar(percent):
    bar_length = 50
    filled_length = int(percent * bar_length)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    print(f'\rProgress: |{bar}| {percent*100:.2f}% ', end='')

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
    #layer_last = Dropout(.2, seed = 777)(layer_4)
    layer_3 = Dense(l3, activation=a3, kernel_initializer='he_uniform' )(layer_2)
    #layer_last = Dropout(.2, seed = 777)(layer_4)
    layer_last = Dense(l4, activation=a4, kernel_initializer='he_uniform' )(layer_3)
    #layer_last = Dropout(.2, seed = 777)(layer_4)
    output_array = Dense(output, activation='softmax', kernel_initializer='he_uniform')(layer_last)

    #Выдает прописной вариант названия активатора
    #keras.activations.serialize(keras.activations.hard_sigmoid)

    model = Model(input_array, output_array)
    return model

def DNN(features):
    input_array = Input(shape=(features,))
    layer_1 = Dense(features, activation='linear', kernel_initializer='he_uniform')(input_array)
    layer_2 = Dense(200, activation='linear', kernel_initializer='he_uniform')(layer_1)
    layer_3 = Dense(400, activation='elu', kernel_initializer='he_uniform')(layer_2)
    layer_4 = Dense(200, activation='linear', kernel_initializer='he_uniform')(layer_3)
    
    
    #layer_drop = Dropout(0.2)(layer_5)
    output_array = Dense(1, activation='elu', kernel_initializer='he_uniform')(layer_4)

    model = Model(input_array, output_array)
    return model

def LogisticRegression(features):
    input_array = Input(shape=(features,))
    layer_1 = Dense(features,activation='tanh',kernel_initializer='he_uniform')(input_array)
    layer_2 = Dense(features,activation='elu',kernel_initializer='he_uniform')(layer_1)
    layer_3 = Dense(features,activation='selu',kernel_initializer='he_uniform')(layer_2)
    layer_3 = Dense(features,activation='relu',kernel_initializer='he_uniform')(layer_3)
    layer_3 = Dropout(.2)(layer_2)
    output_array = Dense(1, activation='sigmoid', kernel_initializer='he_uniform')(layer_3)

    model = Model(input_array,output_array)
    return model

#from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm
#from sklearn.model_selection import RandomizedSearchCV,ShuffleSplit
#from sklearn.utils.fixes import loguniform
from pyod.models.ocsvm import OCSVM
from joblib import dump,load

from sklearn.manifold import TSNE

def dimention_reduction_tsne(X_data,config):

    new_plane_data = TSNE(n_components=config.picture["tSNE"]["n_components"], 
                          learning_rate=config.picture["tSNE"]["learning_rate"], 
                          perplexity=config.picture["tSNE"]["perplexity"], 
                          early_exaggeration=config.picture["tSNE"]["early_exaggeration"],
                          n_iter=config.picture["tSNE"]["n_iter"], 
                          n_iter_without_progress=config.picture["tSNE"]["n_iter_without_progress"], 
                          min_grad_norm=config.picture["tSNE"]["min_grad_norm"],
                          metric=config.picture["tSNE"]["metric"], 
                          init=config.picture["tSNE"]["init"], 
                          verbose=config.picture["tSNE"]["verbose"], 
                          random_state=777, 
                          method=config.picture["tSNE"]["method"], #method='exact'
                          angle=config.picture["tSNE"]["angle"], 
                          n_jobs=-1).fit_transform(X_data)

    return new_plane_data


def outlire(train,data_test,class_weight,name,config):

    from sklearn.preprocessing import MinMaxScaler

    clf = OCSVM(verbose=False,cache_size=10000,max_iter=1000)
    #rng = np.random.default_rng(seed=777)
    #smpl = rng.choice(train,50000)
    if(train.shape[0] < 50000):
        smpl = train.values
    else:
        smpl = train.sample(50000,random_state=777).values
    scaler = MinMaxScaler(feature_range=(-1,1))
    sclr_smpl = scaler.fit_transform(smpl)
    clf.fit(sclr_smpl)
    pd.DataFrame(sclr_smpl).to_csv(f'{config.path_stat}/scaler_values.csv',index=False)
    print(sclr_smpl)
    #clf.fit(smpl)
    dump(clf,f'{config.path_model}_{name}_outlier_clf')
    dump(scaler,f'{config.path_model}_{name}_outlier_scaler')
    scaler = load(f'{config.path_model}_{name}_outlier_scaler')
    print("binary label:\t", clf.labels_)
    print("raw scores:\t", clf.decision_scores_)
    '''
    early_stopping = keras.callbacks.EarlyStopping(
        monitor=config.hyperparam["model_variable"]["early_stopping"]["monitor"], 
        verbose=1,
        patience=2,
        mode=config.hyperparam["model_variable"]["early_stopping"]["mode"],
        restore_best_weights=config.hyperparam["model_variable"]["early_stopping"]["restore_best_weights"])
    '''

    #data_test = pd.DataFrame(data_test)
    data_test_scaler = scaler.transform(data_test.values)
    predict = clf.predict(data_test_scaler)
    print("test data outlier label:\t", predict)
    #print("test data outlier scores:\t",clf.decision_function(data_test.drop(["predict"], axis=1).values))

    print(data_test)
    zero = pd.DataFrame(np.full((5,data_test.shape[1]),-20))
    print(zero)
    zero = scaler.transform(zero)
    print(zero)
    try:
        print("raw outlier scores:\t",clf.decision_function(zero))
        print("binary outlier scores:\t",clf.predict(zero))
    except:
        print("failure")

    #print("data predict one_class_svm:\n",data_test["predict"])

    data_test = data_test[predict == 0] #config.hyperparam["model_variable"]["outlire"]["threshold"]]
    #print(data_test)
    #data_test = data_test.drop(["predict"], axis=1)
    #print(data_test)
    print(data_test)
    
    return np.array(data_test), data_test.index


def redshift_predict(train,label,X_test,y_test,name,config):
    from sklearn.preprocessing import normalize
    features_count = train.shape[1]
    from sklearn.ensemble import RandomForestRegressor

    clf_r = RandomForestRegressor(n_estimators=config.hyperparam["redshift"]["hyperparam"]["n_estimators"],
                                  max_features=config.hyperparam["redshift"]["hyperparam"]["max_features"], 
                                  criterion=config.hyperparam["redshift"]["hyperparam"]["criterion"],
                                  max_depth=config.hyperparam["redshift"]["hyperparam"]["max_depth"],
                                  bootstrap=config.hyperparam["redshift"]["hyperparam"]["bootstrap"],
                                  random_state=777,
                                  n_jobs=-1,
                                  verbose=1)

    clf_r.fit(train.values,label.values)

    predict_red = clf_r.predict(X_test.values)
    
    predict_red = pd.DataFrame(np.array(predict_red), columns=['redshift_pred'])
    y_test = pd.DataFrame(np.array(y_test),columns=['actual_redshift'])
    #X_test = pd.DataFrame(np.array(X_test),columns=get_features(config.features["train"],config))
    predict_red = pd.concat([X_test,predict_red,y_test], axis=1)
    
    predict_red.to_csv(f"{config.path_eval}_{name}_redshift.csv", index=False)

    return clf_r

def redshift_neural_network_API(train,label,X_test,y_test,name,config):
        
    def build_model(hp):
        inputs = keras.Input(shape=(len(get_features(config.features["train"],config)),))
        
        #
        #model.add(layers.Flatten())
        
        x = inputs
        x = Flatten()(x)
        # Tune the number of layers.
        for i in range(hp.Int("num_layers", 1, 3)):
            x = Dense(
                    # Define the hyperparameter.
                    # Tune number of units.
                    units=hp.Int(f"units_{i}", min_value=128, max_value=512, step=128),
                    # Tune the activation function to use.
                    activation=hp.Choice("activation", ["relu", "tanh"]),
                )(x)
            
        # Tune whether to use dropout.
        if hp.Boolean("dropout"):
            x = Dropout(rate=0.25)(x)
        
        outputs = Dense(1, activation="linear")(x)
        model = Model(inputs=inputs, outputs=outputs)
        
        # Define the optimizer learning rate as a hyperparameter.
        learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate), 
            loss=tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error"), 
            metrics=tf.keras.metrics.MeanSquaredError(name="mean_squared_error", dtype=None),
        )
        
        return model

    import keras_tuner

    #hp = keras_tuner.HyperParameters()
    #model = build_model(hp)
    
    tuner = keras_tuner.Hyperband(
        hypermodel=build_model,
        objective="val_accuracy",
        max_epochs=10,
        factor=3,
        directory=f"{config.path_model}/keras_tuner",
        project_name="redshift",
    )

    tuner.search_space_summary()

    tuner.search(train[100000:], label[100000:], epochs=1, validation_data=(train[:100000], label[:100000]))

    # Get the top 2 models.
    models = tuner.get_best_models(num_models=2)
    best_model = models[0]
    # Build the model.
    # Needed for `Sequential` without specified `input_shape`.
    #best_model.build(input_shape=(len(get_features(config.features["train"],config)),))
    best_model.summary()
    #tuner.results_summary()

    # Get the top 2 hyperparameters.
    best_hps = tuner.get_best_hyperparameters(5)
    # Build the model with the best hp.
    model = build_model(best_hps[0])
    # Fit with the entire dataset.
    #x_all = np.concatenate((x_train, x_val))
    #y_all = np.concatenate((y_train, y_val))
    model.fit(x=train, y=label, epochs=20)

    print("model_redshift fited")

    predict_red = model.predict(X_test, batch_size=1024)

    predict_red = pd.DataFrame(np.array(predict_red), columns=['redshift_pred'])
    y_test = pd.DataFrame(np.array(y_test),columns=['actual_redshift'])
    X_test = pd.DataFrame(np.array(X_test),columns=get_features(config.features["train"],config))
    predict_red = pd.concat([X_test,predict_red,y_test], axis=1)
    
    predict_red.to_csv(f"{config.path_predict}_{name}_kerastuner_redshift.csv")

    return model


def model_volume(train,label,X_train,y_train,X_test,y_test,
	model,optimizer,loss,sample_weight,class_weights,num_ep,batch_size,validation_split,
    path_save_eval,name,config):

    early_stopping = keras.callbacks.EarlyStopping(
        monitor=config.hyperparam["model_variable"]["early_stopping"]["monitor"], #"val_auc"
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
    
    history = model.fit(X_train.values, y_train.values,
        epochs=num_ep,
        verbose=1,
        #verbose=0,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stopping],
        class_weight=class_weights,
        #&&&????????????????????????????????????????????????????????????????????
        sample_weight=sample_weight
        )

    print("model fited")

    from graphic import picture_metrics, picture_loss
    #а що як скористатися можливістю збереження через dump?
    if(config.hyperparam["picture"]["metrics"]):
        picture_metrics(history,name,config)
    if(config.hyperparam["picture"]["loss"]):
        picture_loss(history,name,config)

    '''
    import shap
    #normalize data
    
    #X,y = shap.datasets.adult()
    #print(X)
    #print(y)
    X_train = pd.DataFrame(X_train,columns=get_features(config.features["train"],config))
    X_test = pd.DataFrame(X_test,columns=get_features(config.features["train"],config))
    #print(X_train)
    degree = np.sort(np.array([1,2,4,8,16]))
    def f(x):
        pred = model.predict(x)
        #rez = np.zeros(len(pred))
        #rez[:] = pred[:,:].dot(degree)
        rez = np.zeros((len(pred),1))
        rez[:,0] = pred[:,:].dot(degree)
        return rez
    '''
    #print(f(X_train.iloc[:10]))
    #print(X_train.iloc[:10])
    '''
    #y_train_local = y_train[:,:].dot(degree)
    #import xgboost
    #clustering = shap.utils.hclust(X_train, y_train_local)
    #masker = shap.maskers.Partition(X_train, clustering=clustering)
    #explainer = shap.explainers.Exact(f,masker)
    #shap_values = explainer(X_train.iloc[:100])
    #shap.plots.bar(shap_values)
    '''
    '''
    #explainer = shap.explainers.Permutation(f,X_train)
    
    X_test = X_test.sample(200, ignore_index=True)
    #shap_values = explainer(X_test)

    #shap.plots.beeswarm(shap_values)

    explainer = shap.explainers.Exact(f, X_train)
    shap_values = explainer(X_test)
    
    #print(shap_values)
    #print(shap_values[0])
    
    shap.plots.bar(shap_values)
    #shap.plots.waterfall(shap_values[0])
    '''
    #model.evaluate(X_test, y_test, verbose=1)
    #model.summary()
    SaveModel(model,f'{config.path_model}',f'{config.path_weight}',name)
    model = LoadModel(f'{config.path_model}_{name}',f'{config.path_weight}_{name}',optimizer,loss)

    if(config.hyperparam["model_variable"]["metric_culc"] == "test"):
        if(config.hyperparam["model_variable"]["outlire"]["work"]):
            data_test, outlire_index = outlire(X_train,X_test,None,name,config)
            pd_label = pd.DataFrame(np.array(y_test.loc[outlire_index,:]), columns=config.name_class_cls)
        else:
            data_test = X_test.values
            pd_label = pd.DataFrame(np.array(y_test), columns=config.name_class_cls)
        Class = model.predict(data_test, batch_size)
    else:
        Class = model.predict(train, batch_size)
        pd_label = label

    #param from config (columns)
    res = pd.DataFrame(np.array(Class), columns=config.name_class_prob)
    #print(pd_label)
    res = pd.concat([res, pd_label], axis=1)
    
    res.to_csv(f'{path_save_eval}_{name}_prob.csv', index=False)

    #return model

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

    def cust_model(name,X_train,y_train,red_train,red_test,X_test,y_test):
        n = name.split("n")
        l2,l3,l4 = n[1],n[2],n[3]
        a = config.hyperparam["model_variable"]["activation"]
        model = DeepCustomNN_sm(features,l2,l3,l4,a[0],a[1],a[2],len(config.name_class))	
        
        #model1 = 
        model_volume(train,label,X_train,y_train,X_test,y_test,
        model,optimizer,loss,sample_weight,class_weights,num_ep,batch_size,validation_split,
        path_save_eval,f"custom_sm_{name}",config)

        if(config.hyperparam["redshift"]["work"]):
            
            for class_name in config.name_class_cls:
                X_train_temp_class = X_train[y_train[class_name] == 1]
                red_train_temp_class = red_train[y_train[class_name] == 1]
                X_test_temp_class = X_test[y_test[class_name] == 1]
                red_test_temp_class = red_test[y_test[class_name] == 1]
                
            #####
                clf_r = redshift_predict(X_train_temp_class,
                                         red_train_temp_class,
                                         X_test_temp_class,
                                         red_test_temp_class,
                                         f"{name}_{class_name}",config=config)
                dump(clf_r,f'{config.path_model}_custom_sm_{name}_{class_name}_redshift_clf')
            #####
            #model_red = redshift_neural_network_API(X_train,red_train,X_test,red_test,name,config=config)
            #SaveModel(model_red,config.path_model,config.path_weight,f"custom_sm_{name}_redshift")

        #SaveModel(model1,output_path_mod,output_path_weight,f"custom_sm_{name}")
    
    def model_activate(flag,custom_index,X_train,y_train,red_train,red_test,X_test,y_test):
        #param from config 'gpu', 'cpu'
        print(flag)
        if(flag == 'gpu'):
            for name in custom_index:
                cust_model(name,X_train,y_train,red_train,red_test,X_test,y_test)
        #add multithread resolve
        if(flag == 'cpu'):
            import multiprocessing
            MAX_WORKERS = multiprocessing.cpu_count()
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                for name in custom_index:
                    executor.submit(cust_model,name,X_train,y_train,red_train,red_test,X_test,y_test)
    
    #hyperparam from config
    kfold = KFold(n_splits=config.hyperparam["model_variable"]["kfold"], shuffle=False)
    index=0
    for train_index, test_index in kfold.split(train):
        #print("train len ",len(train_index),"test len ",len(test_index))
        #train len  358845 test len  89712
        X_train = train.iloc[train_index]
        y_train = label.iloc[train_index]
        red_train = red_label.iloc[train_index]
    
        X_test = train.iloc[test_index]
        y_test = label.iloc[test_index]
        red_test = red_label.iloc[test_index]
        
        custom_index = []
        #print(index)
        #hyperparam from config
        custom_index.append(make_custom_index(index,config.hyperparam["model_variable"]["neuron_count"]))
        #print(custom_index)
        model_activate(config.flags["system"],custom_index,X_train,y_train,red_train,red_test,X_test,y_test)
        index+=1
    #
    index = '00'
    custom_index = []
    #hyperparam from config
    custom_index.append(make_custom_index(index,config.hyperparam["model_variable"]["neuron_count"]))
    model_activate(config.flags["system"],custom_index,train,label,red_label,red_label,train,label)
    """
    if(open(config.test_path[0])):
        print()
    name = f"custom_sm_{make_custom_index('00',config.hyperparam['model_variable']['neuron_count'])}"
    model = LoadModel(f'{config.path_model}_{name}',f'{config.path_weight}_{name}',optimizer,loss)
    features = get_features(config.features["train"],config)
    for filename in config.test_path:
        test_data = pd.read_csv(filename, header=0, sep=',')
        temp_class = model.predict(test_data[features], batch_size)
        res = pd.DataFrame(np.array(temp_class), columns = config.name_class_prob)
        res = pd.concat([res, test_data[config.name_class_cls]], axis=1)
        res.to_csv(f"{path_save_eval}_{filename.split('/')[-1].split('.')[0]}_prob.csv")
    """

def large_file_prediction(config):
    

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
        print("features create complite")
        del data
        
        #print(data_new.columns.values)
        #print(data_new)
        #data_new.to_csv('/home/lrikozavr/ML_work/test/ml/prediction/data_new.csv')
        if(config.flags['data_preprocessing']['main_sample']['normalize']['work']):
            norms = pd.read_csv(f"{config.path_stat}/{config.name_main_sample}_norms.csv")
            data_new[norms.columns.values] = data_new[norms.columns.values].div(np.array(norms))
            print("normalize complite")

        if(config.flags["prediction"]["outlire"]):
            #outlire_model = LoadModel(f'{config.path_model}_outlire_custom_sm_{name}',f'{config.path_weight}_outlire_custom_sm_{name}',config.hyperparam["optimizer"],"binary_crossentropy")
            outlire_model = load(f'{config.path_model}_custom_sm_{name}_outlier_clf')
            scaler = load(f'{config.path_model}_custom_sm_{name}_outlier_scaler')
            data_new_scaler = scaler.transform(data_new)
            data_new['outlire_prob'] = outlire_model.predict(data_new_scaler)
            del data_new_scaler
            data_new = data_new[data_new['outlire_prob'] == 0].drop(['outlire_prob'], axis=1)
            print("outlier cut complite")

        return data_new
        #get_features(config)

    def ml(output_path_mod,output_path_weight,data,config):
        model = LoadModel(output_path_mod,output_path_weight,config.hyperparam["optimizer"],config.hyperparam["loss"])
        data_col_name = np.concatenate(([config.base[0]],[config.base[1]],get_features(["mags"], config)),axis=0)
        print(data)
        data_temp = deredded(data.replace('null',0.0)[data_col_name].astype(float),config)
        select_index_values = data_temp.index.values
        data = data.iloc[select_index_values].reset_index(drop=True)
        print("deredded")
        del select_index_values
        data_temp = data_temp.reset_index(drop=True)
        print("cut null")
        data = pd.concat([data,data_temp],axis=1)
        #del data_temp
        data_transform = DataTransform(data_temp,config)
        print("Data Transform")
        del data_temp
        predicted = model.predict(data_transform.values, config.hyperparam["batch_size"])
        predicted = pd.DataFrame(np.array(predicted), columns=config.name_class_prob)
        if(config.hyperparam["redshift"]["work"]):
            #rf_model = LoadModel(f"{output_path_mod}_redshift",f"{output_path_weight}_redshift",config.hyperparam["optimizer"],config.hyperparam["loss"])
            rf_predicted = pd.DataFrame()
            for n, class_name in enumerate(config.name_class_prob):
                class_data = data_transform[predicted[class_name] > 0.5]    
                rf_model = load(f'{output_path_mod}_{config.name_class_cls[n]}_redshift_clf')
                rf_class_predicted = rf_model.predict(class_data)
                rf_class_predicted = pd.DataFrame(np.array(rf_class_predicted), columns=['redshift'], index=class_data.index.values)
                rf_predicted = pd.concat([rf_predicted,rf_class_predicted],axis=0)
                del rf_model
                del class_data
        print("predicted")
        data = pd.concat([data,data_transform], axis=1)
        del data_transform
        #
        data = pd.concat([data,predicted], axis=1)
        del predicted
        if(config.hyperparam["redshift"]["work"]):
            #print(data)
            #print(np.sort(rf_predicted.index.values), len(rf_predicted.index.values),rf_predicted)
            #print(rf_predicted.iloc[264347])
            #list_d = np.sort(rf_predicted.index.values)
            #iii = 0
            #for i in list_d:
            #    if(iii != i):
            #        print(iii)
            #        iii = i
            #    iii+=1
            data = pd.concat([data,rf_predicted], axis=1)
            #print(data.iloc[141485])
            #print(data)
            del rf_predicted
        return data
    
    count = config.flags["prediction"]["batch_count"]
    i, index = 0, 1

    global data_mass
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
    name = make_custom_index('00',config.hyperparam["model_variable"]["neuron_count"])
    file_features_null = open(f'{config.prediction_path}_{name}_null_features_index.log','w')
    line_features_null = ['']*(len(config.features["data"]) // 2)
    i=0

    print("begin")
    LINE_COUNT = 0
    for line in f:
        
        LINE_COUNT += 1
        if(LINE_COUNT < count*8.5):
            continue
        if(LINE_COUNT > count*8.6):
            break
        
        if(i // count == index):
            
            index += 1
            #magic
            print(index-1,"start")
            data_mass_temp = pd.DataFrame(data_mass, columns=columns)
            del data_mass
            time.sleep(1)
            data_mass = [[]]*count
            data = ml(f"{config.path_model}_custom_sm_{name}",f"{config.path_weight}_custom_sm_{name}",data_mass_temp,config)
            del data_mass_temp
            data.to_csv(f"{config.path_predict}_{name}_{index-1}.csv", index=False)
            print(index-1,"done")
            del data
            time.sleep(1)

        #print(i - (index-1)*count)
        line_list = list(line.strip('\n').split(","))
        #
        flag_null = 0
        for j in range(len(line_features_null)):
            if (line_list[columns.index(config.features["data"][j*2])] == 'null'):
                line_features_null[j] = config.features["data"][j*2]
                flag_null = 1
            else:
                line_features_null[j] = ''
        if(flag_null):
            file_features_null.write(f"{i}:{','.join(line_features_null)}\n")
            continue
        #
        try:
            float(line_list[columns.index(config.base[0])])
            float(line_list[columns.index(config.base[1])])
        except ValueError:
            print(i," --- line contain bad 'ra', 'dec' value")
            continue
        
        flag_range = 1
        if(len(line_list) == len(columns)):
            for j in range(len(config.features["data"]) // 2):
                if(float(line_list[columns.index(config.features["data"][j*2])]) > config.features["range"][j][0] and float(line_list[columns.index(config.features["data"][j*2])]) < config.features["range"][j][1]):
                    continue
                else:
                    flag_range = 0
                    break            
            if(flag_range):
                data_mass[i - (index-1)*count] = line_list
                i += 1
                loading_progress_bar((i - (index-1)*count)/count)
    #print(data_mass[200][1])
    data_mass = pd.DataFrame(data_mass, columns=columns)
    data_mass = pd.DataFrame(data_mass.head(i - (index-1)*count))
    data = ml(f"{config.path_model}_custom_sm_{name}",f"{config.path_weight}_custom_sm_{name}",data_mass,config)
    data.head(i - (index-1)*count).to_csv(f"{config.path_predict}_{name}_{index}.csv", index=False)
    del data
    file_features_null.close()