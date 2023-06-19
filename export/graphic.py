# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

from main import save_path

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def MCD_plot(d,name):
    count = len(d)
    ar_i = [i for i in range(count)]
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    fig.set_size_inches(10,10)
    ax.scatter(ar_i,d)
    fig.savefig(f"{save_path}/{name}_MCD_distance.png")
    plt.close(fig)    

def plot_loss(ax,history, label, n):
  # Use a log scale on y-axis to show the wide range of values.
  ax.semilogy(history.epoch, history.history['loss'],
               color=colors[n], label='Train ' + label)
  ax.semilogy(history.epoch, history.history['val_loss'],
               color=colors[n], label='Val ' + label,
               linestyle="--")
  ax.xlabel('Epoch')
  ax.ylabel('Loss')

