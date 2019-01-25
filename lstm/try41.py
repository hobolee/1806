# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 08:42:44 2018

@author: DELL
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
for i in range(1, 25):
    if i < 13:
        if i < 10:
            locals()['a'+str(i)]=np.loadtxt(r'D:\1806\1806cal\C40'+str(i)+'-Full.dat', skiprows = 2)
        else:
            locals()['a'+str(i)]=np.loadtxt(r'D:\1806\1806cal\C4'+str(i)+'-Full.dat', skiprows = 2)
    else:
        if i < 22:
            locals()['a'+str(i)]=np.loadtxt(r'D:\1806\DATA FILES\C40'+str(i-12)+'-Full.txt', skiprows = 2)
        else:
            locals()['a'+str(i)]=np.loadtxt(r'D:\1806\DATA FILES\C4'+str(i-12)+'-Full.txt', skiprows = 2)

#cal = a14[:,1]
#surge = a14[:,4]
##t = np.linspace(1, len(surge), len(surge))
#plt.plot(cal)
#plt.plot(surge)

X_train = np.empty((330000,100),dtype="float32")
y_train = np.empty((330000,1),dtype="float32")
X_test = np.empty((30000,100),dtype="float32")
y_test = np.empty((30000,1),dtype="float32")
for i in range(12):
    for j in range(30000):
        if i==11:
            X_test[j,:] = a24[10000+5*j:10500+5*j:5,1]
            y_test[j] = a24[10500+5*j,4] 
        else:
            X_train[30000*i+j,:] = locals()['a'+str(i+13)][10000+5*j:10500+5*j:5,1]
            y_train[30000*i+j,:] = locals()['a'+str(i+13)][10500+5*j,4]
#del a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12
#for i in range(1, 25):
#    del locals()['a'+str(i)]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

import time
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
import keras
def build_model():
    model = Sequential()
    layers = [1, 50, 100, 50, 1]

    model.add(LSTM(
            layers[1],
            input_shape=(None, 1),
            return_sequences=True))
    model.add(Dropout(0.5))

    model.add(LSTM(
            layers[2],
            input_shape=(None, 50),
            return_sequences=True))
    model.add(Dropout(0.5))
    
    model.add(LSTM(
            layers[3],
            return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(
            layers[4]))
    model.add(Activation("linear"))
    start = time.time()
    model.compile(loss="mse", optimizer="adam")
    keras.optimizers.Adam(lr=0.0001)
    print("Compilation Time : ", time.time() - start)
    return model
def run_network(model=None, epochs=0):0
    if model is None:
        model = build_model()
    else:
        model = load_model(model)
        
    try:
        if epochs >0:
            model.fit(
                X_train, y_train,
                batch_size=8192, epochs=epochs, validation_split=0.1)
        predicted = model.predict(X_test)
        predicted = np.reshape(predicted, (predicted.size,))
        predicted_train = model.predict(X_train)
        predicted_train = np.reshape(predicted_train, (predicted_train.size,))
    except KeyboardInterrupt:
        return model, y_test, 0
    try:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(y_test[:, 0])
        plt.plot(predicted[:])
        plt.show()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(y_train[:, 0])
        plt.plot(predicted_train[:])
        plt.show()
    except Exception as e:
        print(str(e))
    if epochs>0:
        model.save('test41a.h5')
    return model, predicted

[mo, pred] = run_network(model='test41a.h5', epochs =10)
#[mo, pred] = run_network(model=None, epochs = 3)