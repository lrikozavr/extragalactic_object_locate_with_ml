import pandas as pd
import numpy as np
import os

from sklearn import svm
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.fixes import loguniform
from sklearn import preprocessing
# import sklearn.metrics as skmetrics
from sklearn.model_selection import ShuffleSplit

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

experiment_name = "svm_balanced"
print(experiment_name)

#----------------- Data: --------------------
train_path = "/home/lrikozavr/ML_work/des_z/ml/data/exgal_main_sample.csv"
general_path = train_path #""

training_data = pd.read_csv(train_path, index_col=0)
general_data = pd.read_csv(general_path, index_col=0)

training_data = training_data.sample(20000,random_state=1)#.reset_index(drop=True)


#----------------- Columns: -----------------
# columns to preserve in the output file:
info_columns = ['Y']

# Features:
features = ['gmag&rmag', 'gmag&imag', 'gmag&zmag', 'gmag&Ymag', 
    'rmag&imag', 'rmag&zmag', 'rmag&Ymag', 
    'imag&zmag', 'imag&Ymag', 
    'zmag&Ymag',
    'gmag','rmag','imag','zmag','Ymag']
output_path = "./results"
#------------------------------------ TRAINING: --------------------------------------
'''
from data_processing import NtoPtoN
index = []
for i in range(20000):
    t = 0
    if (training_data['z'].iloc[i]==0):
        t += 1
    for name in features:
        if(training_data[name].iloc[i]==0):
            t += 1
    if(t==0):
        index.append(i)
training_data = NtoPtoN(training_data,index)
'''
# scale features of the data:
def scale_X_of_the_data(training_X, test_X):
    
    scaler = preprocessing.StandardScaler()
    training_X_transformed = scaler.fit_transform(training_X)
    test_X_transformed = scaler.transform(test_X)
    
    return training_X_transformed, test_X_transformed
train_X, general_X = scale_X_of_the_data(training_data[features], general_data[features])
print('Scaler done')
params = {'C': loguniform(1e0, 1e3),
          'gamma': loguniform(1e-4, 1e-2)}
    

clf = svm.SVR(gamma='scale',
                kernel='rbf',
                cache_size=500)

clf_for_eval = svm.SVR(gamma='scale',
                kernel='rbf',
                cache_size=500)

# create grid search instance:
clf_gs = RandomizedSearchCV(estimator=clf, param_distributions=params, 
                            n_iter=10, scoring='f1', n_jobs=-1, 
                            cv=ShuffleSplit(n_splits=1, test_size=0.2),   
                            refit=True, verbose=0)    #ZMIEŃ
print('RandomizedSearchCV done')
clf_gs.fit(X=train_X, y=training_data["z"])
print('fit done')
def get_gs_results(clf_gs):
    #get grid search results
    
    gs_results = clf_gs.cv_results_
    keys_to_extract = ['mean_test_score', 'std_test_score', 'rank_test_score']
    gs_results_subset = {key: gs_results[key] for key in keys_to_extract}
    gs_results_df = pd.DataFrame(gs_results_subset)
    
    return gs_results_df
# grid search results data frame:
gs_results_df = get_gs_results(clf_gs)

# best parameters from grid search:
best_param_df = pd.DataFrame(clf_gs.best_params_, index=[0])

# evaluation:
clf_for_eval.set_params(**clf_gs.best_params_)

# best model from grid search:
clf_best = clf_gs.best_estimator_
    
# generalization:
general_data["y_pred"] = clf_best.predict(general_X)
#general_data["y_prob_positive_class"] = clf_best.predict_proba(general_X)[:, 1] 
from fuzzy_options import M,D
mass = general_data['z'] - general_data['y_pred']
m = M(mass,mass.shape[0])
d = D(mass,mass.shape[0])
sum = 0
for i in range(mass.shape[0]):
    sum += mass.iloc[i]**2
print('loss---',sum,'M---',m,'D---',d)
#116723.49420318614
print(general_data)
    
print("done.")