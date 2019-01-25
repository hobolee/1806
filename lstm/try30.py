# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 12:23:16 2018

@author: DELL
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from sklearn.preprocessing import scale

def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        if i == interval:
            value = dataset[i - interval] - 0
            diff.append(value/0.075)
        value = dataset[i] - dataset[i - interval]
        diff.append(value/0.075)
    return diff

# invert differenced value
def inverse_difference(history, yhat):
	return yhat + history
def inte(a):
    b = np.zeros((len(a),))
    for i in range(len(a)):
        if i == 0:
            b[i] = a[i]*0.075
        else:
            b[i] = inverse_difference(b[i-1], a[i]*0.075)   
    return b
            

a1 = np.loadtxt(r'D:\1806\DATA FILES\C401-Full.txt')
a2 = np.loadtxt(r'D:\1806\DATA FILES\C403-Full.txt')
a11 = inte(a1[:,-1])[0::3]
a21 = inte(a2[:,-1])[0::3]
a12 = inte(a1[:,2])[0::3]
a22 = inte(a2[:,2])[0::3]
sl = 100
train = np.empty((len(a11)-sl,sl),dtype="float32")
label = np.empty((len(a11)-sl,1),dtype="float32")
train1 = np.empty((len(a11)-sl,sl),dtype="float32")
label1 = np.empty((len(a11)-sl,1),dtype="float32")
test = np.empty((len(a21)-sl,sl),dtype="float32")
test_label = np.empty((len(a21)-sl,1),dtype="float32")
X_train = np.empty((len(a11)-sl,sl),dtype="float32")
y_train = np.empty((len(a11)-sl,1),dtype="float32")
X_test = np.empty((len(a11)-sl,sl),dtype="float32")
y_test = np.empty((len(a11)-sl,1),dtype="float32")
for i in range(len(a11)-sl):
    train[i,:] = a11[0+i:sl+i,]
    label[i] = a12[sl+i] 
for i in range(len(a21)-sl):
    test[i,:] = a21[0+i:sl+i,]
    test_label[i] = a22[sl+i] 
    
y_train = label/2500/35
y_test = test_label/2500/35
X_train = train/2500
X_test = test/2500
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

import time
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
import keras
def build_model():
    model = Sequential()
    layers = [1, 16, 64, 16, 1]

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
#    keras.optimizers.Adam(lr=0.0001)
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
                batch_size=8192, epochs=epochs, validation_split=0.05)
        predicted = model.predict(X_test)
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
        model.save('test30b.h5')
    return model, predicted

#[mo, pred] = run_network(model='test30b.h5', epochs = 0)
[mo, pred] = run_network(model=None, epochs = 1)
