# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 17:43:30 2023

@author: Song Keyu
"""

import numpy as np
import keras
from keras import backend as K

class LSTM_regression:
    
    def __init__(self,
                 LSTM_nodes = 18,
                 input_length=12,
                 channels = 52):
        self.input_length   = input_length
        self.channels       = channels
        self.bce            = keras.losses.BinaryCrossentropy()
        
        input_shape = (self.input_length,self.channels)
        input1 = keras.layers.Input(shape=input_shape)
        cmodel = input1
        cmodel = keras.layers.LSTM(LSTM_nodes, input_shape=input_shape,
                                   go_backwards=False,
                                   return_sequences=False,
                                   activation = "elu",
                                   )(cmodel)
        cmodel = keras.layers.BatchNormalization(axis=-1)(cmodel)
        # cmodel = keras.layers.Dense(80, activation='elu')(cmodel) ## * ##
        cmodel = keras.layers.Dropout(0.2,seed=338)(cmodel)
        
        cmodel = keras.layers.Dense(2, activation='linear')(cmodel)
        # cmodel = keras.layers.Dropout(0.04,seed=338)(cmodel)
        cmodel = keras.layers.Softmax()(cmodel)
        
        self.model = keras.models.Model(inputs=input1,outputs=cmodel)
        self.model.summary()
        # model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
        #               loss='mean_squared_error', metrics=['mse'])
    
    def predict(self, x):
        if self.model is None:
            return None
        return self.model.predict(x)
    
    def fit(self,
            train_x,train_y,
            epochs = 1,
            sample_weight=None):
        opt = keras.optimizers.adam_v2.Adam(learning_rate=5e-3)
        self.model.compile(optimizer=opt,
                      loss='bce', metrics=['bce'])
        self.model.fit(train_x,train_y,
                  epochs=epochs,
                  sample_weight=sample_weight)
        
        
    def __str__(self):
        name = "LSTM_Regression" 
        return name
    
    def save(self,path):
        self.model.save(path)
    def load(self,path):
        self.model=keras.models.load_model(path)
   
    
class LSTM_classifier:
    def __init__(self,
                 LSTM_nodes = 18,
                 input_length=12,
                 channels = 52):
        self.input_length   = input_length
        self.channels       = channels
        self.bce            = keras.losses.BinaryCrossentropy()
        
        input_shape = (self.input_length,self.channels)
        input1 = keras.layers.Input(shape=input_shape)
        cmodel = input1
        cmodel = keras.layers.LSTM(LSTM_nodes, input_shape=input_shape,
                                   go_backwards=False,
                                   return_sequences=False,
                                   activation = "elu",
                                   )(cmodel)
        cmodel = keras.layers.BatchNormalization(axis=-1)(cmodel)
        cmodel = keras.layers.Dropout(0.2,seed=338)(cmodel)
        
        cmodel = keras.layers.Dense(2, activation='linear')(cmodel)
        # cmodel = keras.layers.Dropout(0.04,seed=338)(cmodel)
        cmodel = keras.layers.Softmax()(cmodel)
        
        self.model = keras.models.Model(inputs=input1,outputs=cmodel)
        self.model.summary()
    
    def fit_transform(self,
            train_x,train_y,
            epochs = 60,
            sample_weight=None):
        opt = keras.optimizers.adam_v2.Adam(learning_rate=5e-3)
        self.model.compile(optimizer=opt,
                      loss='bce', metrics=['bce'])
        Y = np.array([train_y,1-train_y]).T
        # print (Y.shape)
        self.model.fit(train_x,Y,
                  epochs=epochs,
                  sample_weight=sample_weight)
        
        return train_x
    
    def transform(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict(X)
    
    def predict(self, X):
        y_proba = self.predict_proba(X)
        y_pred = np.argmax(y_proba, axis=1)
        return y_pred
    
    def save(self,path):
        self.model.save(path)
    def load(self,path):
        self.model=keras.models.load_model(path)