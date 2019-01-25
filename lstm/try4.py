# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 12:47:29 2018

@author: DELL
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
a1=np.loadtxt(r'D:\1806\DATA FILES\C401-Full.txt')
a2=np.loadtxt(r'D:\1806\DATA FILES\C402-Full.txt')
a3=np.loadtxt(r'D:\1806\DATA FILES\C403-Full.txt')
a4=np.loadtxt(r'D:\1806\DATA FILES\C404-Full.txt')
a5=np.loadtxt(r'D:\1806\DATA FILES\C405-Full.txt')

train = np.empty((160000,100),dtype="float32")
label = np.empty((160000,1),dtype="float32")
train1 = np.empty((160000,100),dtype="float32")
label1 = np.empty((160000,1),dtype="float32")
test = np.empty((40000,100),dtype="float32")
testlabel = np.empty((40000,1),dtype="float32")
for i in range(4*40000):
    if i<40000:
        train[i,:] = a1[12000+4*i:12400+4*i:4,-1]
        label[i] = a1[12400+4*i,2]
        test[i,:] = a3[12000+4*i:12400+4*i:4,-1]
        testlabel[i] = a3[12400+4*i,2]
    elif 39999<i<80000:
        train[i,:] = a2[12000+4*(i-40000):12400+4*(i-40000):4,-1]
        label[i] = a2[12400+4*(i-40000),2]
    elif 79999<i<120000:
        train[i,:] = a3[12000+4*(i-80000):12400+4*(i-80000):4,-1]
        label[i] = a4[12400+4*(i-80000),2]
    else:
        train[i,:] = a4[12000+4*(i-120000):12400+4*(i-120000):4,-1]
        label[i] = a4[12400+4*(i-120000),2]        
    

iii = [ii for ii in range(160000)]
np.random.shuffle(iii)
for i in range(160000):
    train1[i,:] = train[iii[i],:]
    label1[i] = label[iii[i]]
X_train = train1
y_train = label1
X_test = test
y_test = testlabel
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
                batch_size=2048, nb_epoch=epochs, validation_split=0.05)
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
        model.save('test3.h5')
    return model, predicted

[mo, pred] = run_network(model='test4b.h5', epochs = 0)