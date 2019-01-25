# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 21:19:41 2018

@author: DELL
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
for i in range(1, 81):
    if i < 41:
        if i < 10:
            locals()['a'+str(i)]=np.loadtxt(r'D:\1806\1806cal\C10'+str(i)+'-Full.dat', skiprows = 2)
        elif i < 17:
            locals()['a'+str(i)]=np.loadtxt(r'D:\1806\1806cal\C1'+str(i)+'-Full.dat', skiprows = 2)
        elif i < 26:
            locals()['a'+str(i)]=np.loadtxt(r'D:\1806\1806cal\C40'+str(i-16)+'-Full.dat', skiprows = 2)
        else:
            locals()['a'+str(i)]=np.loadtxt(r'D:\1806\1806cal\C4'+str(i-16)+'-Full.dat', skiprows = 2)
    else:
        if i < 50:
            locals()['b'+str(i-40)]=np.loadtxt(r'D:\1806\DATA FILES\C10'+str(i-40)+'-Full.txt', skiprows = 2)
        elif i < 57:
            locals()['b'+str(i-40)]=np.loadtxt(r'D:\1806\DATA FILES\C1'+str(i-40)+'-Full.txt', skiprows = 2)
        elif i < 66:
            locals()['b'+str(i-40)]=np.loadtxt(r'D:\1806\DATA FILES\C40'+str(i-56)+'-Full.txt', skiprows = 2)
        else:
            locals()['b'+str(i-40)]=np.loadtxt(r'D:\1806\DATA FILES\C4'+str(i-56)+'-Full.txt', skiprows = 2)

#cal = a17[:,1]
#surge = b17[:,2]
#t = np.linspace(1, len(surge), len(surge))
#plt.plot(t, cal[0:len(surge)], t, surge)

X_train = np.empty((1657500,75),dtype="float32")
y_train = np.empty((1657500,1),dtype="float32")
X_test = np.empty((42500,75),dtype="float32")
y_test = np.empty((42500,1),dtype="float32")
for i in range(1, 41):
    for j in range(42500):
        if i==40:
            X_test[j,:] = a40[0+4*j:300+4*j:4,1]
            y_test[j] = b40[300+4*j,2] 
        else:
            if i > 16:
                X_train[42500*(i-1)+j,:] = locals()['a'+str(i+1)][0+4*j:300+4*j:4,1]
            else:
                X_train[42500*(i-1)+j,:] = locals()['a'+str(i+1)][0+2*j:150+2*j:2,1]
            y_train[42500*(i-1)+j,:] = locals()['b'+str(i+1)][300+4*j,2]
#for i in range(1, 40):
#    del locals()['a'+str(i)]
#    del locals()['b'+str(i)] 
            
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
def run_network(model=None, epochs=0):
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
    if epochs>0:
        model.save('test37a.h5')
    return model, predicted

[mo, pred] = run_network(model='test37a.h5', epochs =20)
#[mo, pred] = run_network(model=None, epochs = 1)