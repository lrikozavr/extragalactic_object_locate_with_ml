# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

import seaborn as sns

from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

from main import save_path

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

def plot_loss(ax, history, label, n):
  # Use a log scale on y-axis to show the wide range of values.
  ax.semilogy(history.epoch, history.history['loss'],
               color=colors[n], label='Train ' + label)
  ax.semilogy(history.epoch, history.history['val_loss'],
               color=colors[n], label='Val ' + label,
               linestyle="--")
  ax.xlabel('Epoch')
  ax.ylabel('Loss')

def plot_cm(name, labels, predictions, p=0.5):
  cm = confusion_matrix(labels, predictions > p)
  fig = plt.figure(figsize=(5,5))
  sns.heatmap(cm, annot=True, fmt="d")
  plt.title('Confusion matrix, threshold: {:.2f}'.format(p))
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')
  fig.savefig(f'{save_path}/{name}_Confusion_matrix.png')
  plt.close(fig)

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

def data_phot_hist(path_sample = '/home/lrikozavr/ML_work/des_pro/ml/data', save_path = '/home/lrikozavr/ML_work/des_pro/ml/pictures/hist'):
    features = ['gmag&rmag', 'gmag&imag', 'gmag&zmag', 'gmag&Ymag', 
    'rmag&imag', 'rmag&zmag', 'rmag&Ymag', 
    'imag&zmag', 'imag&Ymag', 
    'zmag&Ymag',
    'gmag','rmag','imag','zmag','Ymag']
    class_name_mass = ['star','qso','gal']
    from grafics import Hist1
    for name in class_name_mass:
        data = pd.read_csv(f"{path_sample}/{name}_main_sample.csv", header=0, sep=',')
        for name_phot in features:
            Hist1(data[name_phot],save_path,f'{name}_{name_phot}',name_phot)