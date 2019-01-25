# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 17:03:25 2018

@author: DELL
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
a1=np.loadtxt(r'D:\1806\DATA FILES\C111-Full.txt')
#a1=a1[12000:172400,1:3]
#plt.plot(a1[:,0],a1[:,1])
#a2=np.loadtxt(r'D:\1806\DATA FILES\C110-Full.txt')
#a2=a2[12000:174000,1:3]
#plt.plot(a2[:,0],a2[:,1])
#a3=np.loadtxt(r'D:\1806\DATA FILES\C111-Full.txt')
#a3=a3[12000:174000,1:3]
#plt.plot(a3[:,0],a3[:,1])
#a4=np.loadtxt(r'D:\1806\DATA FILES\C112-Full.txt')
#a4=a4[12000:174000,1:3]
#plt.plot(a4[:,0],a4[:,1])

train = np.empty((16000,100),dtype="float32")
label = np.empty((16000,1),dtype="float32")
train1 = np.empty((16000,100),dtype="float32")
label1 = np.empty((16000,1),dtype="float32")
for i in range(16000):
    train[i,:] = a1[12000+i:12100+i,-1]
    label[i] = a1[12100+i,2]
    

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
#plt.plot(train[:,0])
#plt.plot(label[:,0])
#plt.plot(train[:,0],label[:,0])
#plt.plot(train1[:,0])
#plt.plot(label1[:,0])
#plt.figure()
#plt.plot(train1[:,0],label1[:,0])


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
def run_network(model=None):
    epochs = 300

    if model is None:
        model = build_model()
        
    try:
        model.fit(
            X_train, y_train,
            batch_size=512, nb_epoch=epochs, validation_split=0.05)
        predicted = model.predict(X_test)
        predicted = np.reshape(predicted, (predicted.size,))
    except KeyboardInterrupt:
#        print('Training duration (s) : ', time.time() - global_start_time)
        return model, y_test, 0
    try:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(y_test[:, 0])
        plt.plot(predicted[:])
        plt.show()
    except Exception as e:
        print(str(e))
#    print('Training duration (s) : ', time.time() - global_start_time)
    model.save('test1')
    return model, y_test, predicted


[mo, y_t, pred] = run_network()