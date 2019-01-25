# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 10:10:45 2018

@author: DELL
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
a1=np.loadtxt(r'D:\1806\DATA FILES\C111-Full.txt')

train = np.empty((8000,100),dtype="float32")
label = np.empty((8000,1),dtype="float32")
train1 = np.empty((8000,100),dtype="float32")
label1 = np.empty((8000,1),dtype="float32")
for i in range(8000):
    train[i,:] = a1[12000+2*i:12200+2*i:2,-1]
    label[i] = a1[12200+2*i,2]
    

iii = [ii for ii in range(16000)]
np.random.shuffle(iii)
for i in range(16000):
    train1[i,:] = train[iii[i],:]
    label1[i] = label[iii[i]]
X_train = train1[0:15000,:]
y_train = label1[0:15000,:]
X_test = train1[15000:,:]
y_test = label1[15000:,:]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

import time
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
def build_model():
    model = Sequential()
    layers = [1, 50, 100, 1]

    model.add(LSTM(
            layers[1],
            input_shape=(None, 1),
            return_sequences=True))
    model.add(Dropout(0.2))
    
    model.add(LSTM(
            layers[2],
            return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(
            layers[3]))
    model.add(Activation("linear"))
    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("Compilation Time : ", time.time() - start)
    return model
def run_network(model=None, epochs=0):
    if model is None:
        model = build_model()
    else:
        model = load_model(model)
        
    try:
        if epochs >0:
            model.fit(
                X_train, y_train,
                batch_size=512, nb_epoch=epochs, validation_split=0.05)
        predicted = model.predict(X_test)
        predicted = np.reshape(predicted, (predicted.size,))
    except KeyboardInterrupt:
        return model, y_test, 0
    try:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(y_test[:, 0])
        plt.plot(predicted[:])
        plt.show()
    except Exception as e:
        print(str(e))
    model.save('test3.h5')
    return model, predicted

[mo, pred] = run_network(epochs = 300)