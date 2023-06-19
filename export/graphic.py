# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def plot_loss(ax,history, label, n):
  # Use a log scale on y-axis to show the wide range of values.
  ax.semilogy(history.epoch, history.history['loss'],
               color=colors[n], label='Train ' + label)
  ax.semilogy(history.epoch, history.history['val_loss'],
               color=colors[n], label='Val ' + label,
               linestyle="--")
  ax.xlabel('Epoch')
  ax.ylabel('Loss')

