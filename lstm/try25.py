# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 22:06:46 2018

@author: DELL
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 15:40:53 2018

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
a6=np.loadtxt(r'D:\1806\DATA FILES\C406-Full.txt')
a7=np.loadtxt(r'D:\1806\DATA FILES\C407-Full.txt')
a8=np.loadtxt(r'D:\1806\DATA FILES\C408-Full.txt')
a9=np.loadtxt(r'D:\1806\DATA FILES\C409-Full.txt')
a10=np.loadtxt(r'D:\1806\DATA FILES\C410-Full.txt')
a11=np.loadtxt(r'D:\1806\DATA FILES\C411-Full.txt')


train = np.empty((1700000,100),dtype="float32")
label = np.empty((1700000,1),dtype="float32")
X_train = np.empty((1700000,100),dtype="float32")
y_train = np.empty((1700000,1),dtype="float32")
X_test = np.empty((170000,100),dtype="float32")
y_test = np.empty((170000,1),dtype="float32")
for i in range(170000):
    train[10*i,:] = a1[0+i:100+i,-1]
    label[10*i] = a1[100+i,2] 
    X_test[i,:] = a6[0+i:100+i,-1]
    y_test[i] = a6[100+i,2] 
    train[10*i+1,:] = a2[0+i:100+i,-1]
    label[10*i+1] = a2[100+i,2]    
    train[10*i+2,:] = a3[0+i:100+i,-1]
    label[10*i+2] = a3[100+i,2] 
    train[10*i+3,:] = a4[0+i:100+i,-1]
    label[10*i+3] = a4[100+i,2]  
    train[10*i+4,:] = a5[0+i:100+i,-1]
    label[10*i+4] = a5[100+i,2] 
    train[10*i+5,:] = a7[0+i:100+i,-1]
    label[10*i+5] = a7[100+i,2]    
    train[10*i+6,:] = a8[0+i:100+i,-1]
    label[10*i+6] = a8[100+i,2] 
    train[10*i+7,:] = a9[0+i:100+i,-1]
    label[10*i+7] = a9[100+i,2]  
    train[10*i+8,:] = a10[0+i:100+i,-1]
    label[10*i+8] = a10[100+i,2] 
    train[10*i+9,:] = a11[0+i:100+i,-1]
    label[10*i+9] = a11[100+i,2]    
del a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11
#iii = [ii for ii in range(1700000)]
#np.random.shuffle(iii)
#for i in range(1700000):
#    X_train[i,:] = train[iii[i],:]
#    y_train[i] = label[iii[i]]
X_train = train
y_train = label
del train,label
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

import time
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
import keras
def build_model():
    model = Sequential()
    layers = [1, 64, 128, 64, 1]

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
                batch_size=8192, nb_epoch=epochs, validation_split=0.05)
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
        model.save('test25b.h5')
    return model, predicted

[mo, pred] = run_network(model='test25b.h5', epochs = 0)
#[mo, pred] = run_network(model=None, epochs = 20)