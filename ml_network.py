#!/home/kiril/python_env_iron_ment/new_proj/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import os
import math

from keras.layers import Input, Dense, Dropout
from keras.layers import experimental, Conv2D, SeparableConv2D, MaxPooling2D, GlobalAveragePooling2D, Activation, add
from keras.models import Model
from keras.layers.normalization import BatchNormalization
import keras

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,KFold
import tensorflow as tf
from keras import backend as K
import numpy as np

import sklearn.metrics as skmetrics

def DeepLayerNN(features):
    input_array = Input(shape=(features,))
    layer_1 = Dense(features, activation='linear', kernel_initializer='he_uniform')(input_array)
    layer_2 = Dense(16, activation="softsign" )(layer_1)
    layer_3 = Dense(8, activation="tanh" )(layer_2)
    layer_last = Dropout(.2)(layer_3)
    output_array = Dense(1, activation='swish', kernel_initializer='he_uniform')(layer_last)

    #Выдает прописной вариант названия активатора
    #keras.activations.serialize(keras.activations.hard_sigmoid)

    model = Model(input_array, output_array)
    return model
    
def LinearDeepNN(features):
    input_img = Input(shape=(features,))
    layer_1 = Dense(64, activation='linear', kernel_initializer='he_uniform' )(input_img)
    layer_2 = Dense(32, activation='linear', kernel_initializer='he_uniform' )(layer_1)
    layer_3 = Dense(16, activation='linear', kernel_initializer='he_uniform' )(layer_2)
    layer_4 = Dense(8, activation='linear', kernel_initializer='he_uniform' )(layer_3)
    layer_5 = Dense(4, activation='linear', kernel_initializer='he_uniform' )(layer_4)
    Label = Dense(1,activation='linear', kernel_initializer='he_uniform' )(layer_5)
    model = Model(input_img, Label)
    return model