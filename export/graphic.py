# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

import seaborn as sns

from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

#from main import save_path
from network import LoadModel, make_custom_index


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def MCD_plot(name,d):
    count = len(d)
    ar_i = [i for i in range(count)]
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    fig.set_size_inches(10,10)
    ax.scatter(ar_i, d, color = colors[0])
    fig.savefig(f"{save_path}/{name}_MCD_distance.png")
    plt.close(fig)    

def picture_cm(config):
  def plot_cm(index,save_name):
    name = make_custom_index(index,config.hyperparam["model_variable"]["neuron_count"])
    data = pd.read_csv(f'{config.path_eval}_custom_sm_{name}_prob.csv', header=0, sep=",")
    
    y = np.argmax(data[config.name_class_cls], axis=1).tolist()
    y_prob = np.argmax(data[config.name_class_cls_prob], axis=1).tolist()

    cm = confusion_matrix(y, y_prob)
    fig = plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    fig.savefig(f'{config.path_pic}/{save_name}_Confusion_matrix.png')
    plt.close(fig)

  for i in range(config.hyperparam["model_variable"]["kfold"]):
    plot_cm(i,f"{config.sample_name}_{i}_kfold")
  #
  plot_cm('00',f"{config.sample_name}_main")
  #

def plot_metrics(axs, name, history):
  metrics = ['loss', 'auc', 'precision', 'recall']
  #fig, axs = plt.subplots(2,2)

  for n, metric in enumerate(metrics):
    name = metric.replace("_"," ").capitalize()
    i, j = n//2, n%2
    axs[i,j].plot(history.epoch, history.history[metric], color=colors[0], label='Train_' + name)
    axs[i,j].plot(history.epoch, history.history['val_'+metric],
             color=colors[0], linestyle="--", label='Val_' + name)
    axs[i,j].xlabel('Epoch')
    axs[i,j].ylabel(name)
    if metric == 'loss':
      axs[i,j].ylim([0, plt.ylim()[1]])
    elif metric == 'auc':
      axs[i,j].ylim([0.8,1])
    else:
      axs[i,j].ylim([0,1])

    axs[i,j].legend()

def data_hist(config,data=None):
    
    def Hist1(ax,x,mag,label, **kwargs):
        ax.set_xlabel(mag,fontsize=40)
        ax.set_ylabel("count",fontsize=40)
        ax.tick_params(axis='x', labelsize=30)
        ax.tick_params(axis='y', labelsize=30)
        #ax.set_title(name,fontsize = 50)
        ax.hist(x,bins=200, label=label,**kwargs)

    if data == None:
        data = pd.DataFrame()
        for class_name in config.name_class:
            data_temp = pd.read_csv(f"{config.path_ml_data}/{class_name}_main_sample.csv", header=0, sep=',')       
            data = pd.concat([data,data_temp], ignore_index=True)
    elif():
        data_mass = []
        for class_name in config.name_class:
            data_temp = pd.read_csv(f"{config.path_ml_data}/{class_name}_main_sample.csv", header=0, sep=',')       
            data_mass.append(data_temp)
    else:     
        data = data.drop(config.name_class_cls, axis = 1)

    columns = data_mass[0].drop(config.base, axis=1).columns.values

    if():
        for col in columns:
            for i, class_name in enumerate(config.name_class):
                fig=plt.figure()
                ax = fig.add_subplot(1,1,1)
                fig.suptitle(col, fontsize=50)       
                Hist1(ax,data_mass[i][col],col,class_name, histtype='step', stacked=False, fill=True)
                ax.legend()
                fig.set_size_inches(30,20)
                fig.savefig(f"{config.path_pic}/{class_name}_{col}_hist.png")
                plt.close(fig)
    elif():
        for col in columns:
            fig=plt.figure()
            ax = fig.add_subplot(1,1,1)
            fig.suptitle(col, fontsize=50)       
            for i, class_name in enumerate(config.name_class):
                Hist1(ax,data_mass[i][col],col,class_name, histtype='step', stacked=False, fill=False)
            ax.legend()
            fig.set_size_inches(30,20)
            fig.savefig(f"{config.path_pic}/{col}_hist.png")
            plt.close(fig)

def picture_loss(optimizer,loss,config):

  def plot_loss(ax, history, label, n):
    # Use a log scale on y-axis to show the wide range of values.
    ax.semilogy(history.epoch, history.history['loss'],
                color=colors[n], label='Train ' + label)
    ax.semilogy(history.epoch, history.history['val_loss'],
                color=colors[n], label='Val ' + label,
                linestyle="--")
    ax.xlabel('Epoch')
    ax.ylabel('Loss')

  fig = plt.figure()
  ax = fig.add_subplot(1,1,1)
  
  for i in range(config.hyperparam["model_variable"]["kfold"]):
    name = make_custom_index(i,config.hyperparam["model_variable"]["neuron_count"])
    model = LoadModel(f"{config.path_model}_custom_sm_{name}",f"{config.path_weights}_custom_sm_{name}",optimizer,loss)
    plot_loss(ax,model,f"{config.sample_name}_{i}",i)
  ax.set_label(config.sample_name)
  ax.legend()
  fig.savefig(f'{config.path_pic}/{config.sample_name}_kfold_summary_loss.png')
  del fig

def picture_roc_prc(config):  

  def plot_roc(ax, name, labels, predictions, **kwargs):
    fp, tp, _ = roc_curve(labels, predictions)

    #fig = plt.figure(figsize=(5,5))
    ax.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
    ax.xlabel('False positives [%]')
    ax.ylabel('True positives [%]')
    ax.xlim([-0.5,20])
    ax.ylim([80,100.5])
    ax.grid(True)
    #ax = plt.gca()
    ax.set_aspect('equal')
    #fig.savefig(f'{save_path}/{name}_ROC.png')
    #plt.close(fig)

  def plot_prc(ax, name, labels, predictions, **kwargs):
      precision, recall, _ = precision_recall_curve(labels, predictions)

      ax.plot(precision, recall, label=name, linewidth=2, **kwargs)
      ax.xlabel('Precision')
      ax.ylabel('Recall')
      ax.grid(True)
      #ax = plt.gca()
      ax.set_aspect('equal')

  data_mass = []
  for i in range(config.hyperparam["model_variable"]["kfold"]):
    name = make_custom_index(i,config.hyperparam["model_variable"]["neuron_count"])
    data = pd.read_csv(f'{config.path_eval}_custom_sm_{name}_prob.csv', header=0, sep=",")
    data_mass.append(data)
  
  for key in config.picture["roc_prc"]:
    match key:
        case 1:
            for class_name in config.name_class:
                fig = plt.figure()
                ax_prc = fig.add_subplot(1,2,2)
                ax_roc = fig.add_subplot(1,2,1)
                for i in range(config.hyperparam["model_variable"]["kfold"]):
                    plot_prc(ax_prc,f"{i}_kfold",data_mass[i][f'{class_name}_cls'],data_mass[i][f'{class_name}_cls_prob'])
                    plot_roc(ax_roc,f"{i}_kfold",data_mass[i][f'{class_name}_cls'],data_mass[i][f'{class_name}_cls_prob'])
                fig.set_label(f"{config.sample_name}_{class_name}")
                fig.legend()
                fig.savefig(f'{config.path_pic}/{config.sample_name}_{class_name}_kfold_summary_roc_prc.png')
        case 2:
            fig = plt.figure()
            ax_prc = fig.add_subplot(1,2,2)
            ax_roc = fig.add_subplot(1,2,1)
            for class_name in config.name_class:
                for i in range(config.hyperparam["model_variable"]["kfold"]):
                    plot_prc(ax_prc,f"{i}_{class_name}",data_mass[i][f'{class_name}_cls'],data_mass[i][f'{class_name}_cls_prob'])
                    plot_roc(ax_roc,f"{i}_{class_name}",data_mass[i][f'{class_name}_cls'],data_mass[i][f'{class_name}_cls_prob'])
            fig.set_label(f"{config.sample_name}")
            fig.legend()
            fig.savefig(f'{config.path_pic}/{config.sample_name}_class_kfold_summary_roc_prc.png')
        case 3:
            for i in range(config.hyperparam["model_variable"]["kfold"]):
                fig = plt.figure()
                ax_prc = fig.add_subplot(1,2,2)
                ax_roc = fig.add_subplot(1,2,1)
                for class_name in config.name_class:
                    plot_prc(ax_prc,class_name,data_mass[i][f'{class_name}_cls'],data_mass[i][f'{class_name}_cls_prob'])
                    plot_roc(ax_roc,class_name,data_mass[i][f'{class_name}_cls'],data_mass[i][f'{class_name}_cls_prob'])
                fig.set_label(f"{config.sample_name}_{i}_kfold")
                fig.legend()
                fig.savefig(f'{config.path_pic}/{config.sample_name}_{i}_kfold_class_summary_roc_prc.png')
        case 4:
            for i in range(config.hyperparam["model_variable"]["kfold"]):
                for class_name in config.name_class:
                    fig = plt.figure()
                    ax_prc = fig.add_subplot(1,2,2)
                    ax_roc = fig.add_subplot(1,2,1)
                    plot_prc(ax_prc,f"{class_name}_{i}_kfold",data_mass[i][f'{class_name}_cls'],data_mass[i][f'{class_name}_cls_prob'])
                    plot_roc(ax_roc,f"{class_name}_{i}_kfold",data_mass[i][f'{class_name}_cls'],data_mass[i][f'{class_name}_cls_prob'])
                    fig.set_label(f"{config.sample_name}_{class_name}_{i}_kfold")
                    fig.legend()
                    fig.savefig(f'{config.path_pic}/{config.sample_name}_{i}_kfold_{class_name}_summary_roc_prc.png')
