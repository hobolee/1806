# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 21:59:49 2018

@author: DELL
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM

a1=np.loadtxt(r'D:\1806\DATA FILES\C111-Full.txt')
a2=np.loadtxt(r'D:\1806\DATA FILES\C110-Full.txt')

train = np.empty((160000,100),dtype="float32")
label = np.empty((160000,1),dtype="float32")
train1 = np.empty((160000,100),dtype="float32")
label1 = np.empty((160000,1),dtype="float32")
test = np.empty((160000,100),dtype="float32")
testlabel = np.empty((160000,1),dtype="float32")
for i in range(160000):
    train[i,:] = a1[12000+i:12100+i,-1]
    label[i] = a1[12100+i,2]
    test[i,:] = a1[12000+i:12100+i,-1]
    testlabel[i] = a1[12100+i,2]

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
    if model is None:
        model = build_model()
        model.load_weights('test2.h5')
        predicted = model.predict(X_test)
        predicted = np.reshape(predicted, (predicted.size,))
    try:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(y_test[:, 0])
        plt.plot(predicted[:])
        plt.show()
    except Exception as e:
        print(str(e))
#    print('Training duration (s) : ', time.time() - global_start_time)
    return model, y_test, predicted

[mo, y_t, pred] = run_network()