# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math

#Середнє, дисперсія і т.д. для всіх значень метрик отриманих після Kfold
from data_process import M, D
from network import make_custom_index

def metric_statistic(config):
    print("make metrics stat")

    thr = lambda x: 1 if x >= 0.5 else 0

    def eval(y_prediction,y_actual):
        count = 0
        TP, FP, TN, FN = 0,0,0,0
        #Y = 0
        Y = y_prediction.apply(thr)
        TP = Y[(1 == y_actual) & (Y == 1)].shape[0]
        FP = Y[(0 == y_actual) & (Y == 1)].shape[0]
        TN = Y[(0 == y_actual) & (Y == 0)].shape[0]
        FN = Y[(1 == y_actual) & (Y == 0)].shape[0]
        '''
        for i in range(y_prediction.shape[0]):
            Y = thr(y_prediction[i],0.5)
            if(Y==y_actual[i]):
                count+=1
            if(Y==1):
                if(Y==y_actual[i]):
                    TP += 1
                else:
                    FP += 1
            if(Y==0):
                if(Y==y_actual[i]):
                    TN += 1
                else:
                    FN += 1
        '''
        try:
            Acc = Y[Y == y_actual].shape[0]/Y.shape[0]
        except:
            print("Acc division by zero")
            Acc = 999.0
        try:        
            pur_a = TP/(TP+FP)
        except:
            print("pur_a division by zero")
            pur_a = 999.0
        try:
            com_a = TP/(TP+FN)
        except:
            print("com_a division by zero")
            com_a = 999.0
        try:
            f1 = 2*TP/(2*TP+FP+FN)
        except:
            print("f1 division by zero")
            f1 = 999.0
        try:
            fpr = FP/(TN+FN)
        except:
            print("fpr division by zero")
            fpr = 999.0
        try:
            tnr = TN/(TN+FN)
        except:
            print("tnr division by zero")
            tnr = 999.0
        try:
            bAcc = (TP/(TP+FP)+TN/(TN+FN))/2.
        except:
            print("bAcc division by zero")
            bAcc = 999.0
        try:
            k = 2*(TP*TN-FN*FP)/((TP+FP)*(FP+TN)+(TP+FN)*(FN+TN))
        except:
            print("k division by zero")
            k = 999.0
        try:
            mcc = (TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        except:
            print("mcc division by zero")
            mcc = 999.0
        try:
            BinBs = (FP+FN)/(TP+FP+FN+TN)
        except:
            print("BinBs division by zero")
            BinBs = 999.0

        #print(np.array([Acc,pur_a,com_a,f1,fpr,tnr,bAcc,k,mcc,BinBs]))
        ev = pd.DataFrame([np.array([Acc,pur_a,com_a,f1,fpr,tnr,bAcc,k,mcc,BinBs])], 
        columns=['Accuracy','Purity','Completness','F1',
        'FPR','TNR','bACC','K','MCC','BinaryBS'])

        return ev

    def metric_stat(data):
        res = pd.DataFrame()
        for name in data.columns.values:
            average = data[name].mean()
            dispersion = data[name].std()
            max = data[name].max()
            min = data[name].min()
            #print([average,dispersion,max,min])
            df = pd.DataFrame(np.array([average,dispersion,max,min]), columns=[name], index=["mean","std","max","min"]).transpose()
            #print(df)
            res = pd.concat([res,df], axis=0)
        #res = res.transpose()
        return res

    ev_data = [pd.DataFrame() for i in range(len(config.name_class))]
    for i in range(config.hyperparam["model_variable"]["kfold"]):
        name = make_custom_index(i,config.hyperparam["model_variable"]["neuron_count"])
        data = pd.read_csv(f"{config.path_eval}_custom_sm_{name}_prob.csv", header=0, sep=",")
        for n in range(len(config.name_class)):
            ev_data_temp = eval(data.loc[data[config.name_class_cls[n]] == 1,config.name_class_prob[n]],
                                data.loc[data[config.name_class_cls[n]] == 1,config.name_class_cls[n]])
            ev_data[n] = pd.concat([ev_data[n],ev_data_temp],ignore_index=True)
    
    for i, name in enumerate(config.name_class):
        metric_data = metric_stat(ev_data[i])
        metric_data.to_csv(f"{config.path_stat}/{config.name_sample}_{name}_kfold_summary_metric_statistic.csv")

    del ev_data
    
    #main
    name = make_custom_index('00',config.hyperparam["model_variable"]["neuron_count"])
    data = pd.read_csv(f"{config.path_eval}_custom_sm_{name}_prob.csv", header=0, sep=",")
    for n, name in enumerate(config.name_class):
        ev_data_temp = eval(data.loc[data[config.name_class_cls[n]] == 1,config.name_class_prob[n]],
                            data.loc[data[config.name_class_cls[n]] == 1,config.name_class_cls[n]])
        ev_data_temp.to_csv(f"{config.path_stat}/{config.name_sample}_{name}_main_metric.csv")

    del data

def metric_sklearn(y_actual, y_prediction, threshold = 0.5):

    thr = lambda x: 1 if x >= threshold else 0

    from sklearn import metrics as m

    Y = np.apply_along_axis(thr,0,y_prediction)

    skmetrics = lambda x,y: pd.DataFrame(np.array([m.accuracy_score(x,y),
                                        m.precision_score(x,y),
                                        m.recall_score(x,y),
                                        m.f1_score(x,y),
                                        m.balanced_accuracy_score(x,y),
                                        m.cohen_kappa_score(x,y),
                                        m.matthews_corrcoef(x,y),
                                        m.roc_auc_score(x,y)]),
                            columns=['accuracy','precision','recall','f1',
                                        'bacc','k','mcc','roc'])

    fpr, tpr, thresholds = m.roc_curve(y_actual,y_prediction)
    roc_curve_df = pd.DataFrame({"fpr": fpr, "tpr": tpr,
									"thresholds": thresholds})

    precision, recall, thresholds = m.precision_recall_curve(y_actual,y_prediction)
    pr_curve_df = pd.DataFrame({"precision": precision, "recall": recall, 
                                    "thresholds": np.append(thresholds, 1)})

    return skmetrics(y_actual,Y), roc_curve_df, pr_curve_df

