# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 22:14:59 2018

@author: DELL
"""
import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
#time, wave.C1, surge, sway, heave, roll, pitch, yaw
for i in range(1, 41):
    if i < 41:
        if i < 10:
            locals()['a' + str(i)] = np.loadtxt(r'/lustre/home/naolmy/hobolee/1806/lstm/data_aligned/C30' + str(i) + '.txt',
                                                skiprows=0)
        elif i < 17:
            locals()['a' + str(i)] = np.loadtxt(r'/lustre/home/naolmy/hobolee/1806/lstm/data_aligned/C3' + str(i) + '.txt',
                                                skiprows=0)
        elif i < 26:
            locals()['a' + str(i)] = np.loadtxt(
                r'/lustre/home/naolmy/hobolee/1806/lstm/data_aligned/C50' + str(i - 16) + '.txt', skiprows=0)
        else:
            locals()['a' + str(i)] = np.loadtxt(r'/lustre/home/naolmy/hobolee/1806/lstm/data_aligned/C5' + str(i - 16) + '.txt',
                                                skiprows=0)
gap = 8
timeSteps = 75
for whichMotion in range(2, 8):
    # whichMotion = 3
    X_train = np.empty((39*170000//gap, timeSteps), dtype="float32")
    y_train = np.empty((39*170000//gap, 1), dtype="float32")
    X_test = np.empty((170000//gap, timeSteps), dtype="float32")
    y_test = np.empty((170000//gap, 1), dtype="float32")
    for i in range(1, 41):
        for j in range(170000//gap):
            if i == 40:
                X_test[j, :] = a10[0 + gap * j:gap*timeSteps + gap * j:gap, 1]
                y_test[j] = a10[gap*timeSteps + gap * j, whichMotion]
            else:
                X_train[170000//gap * (i - 1) + j, :] = locals()['a' + str(i + 1)][0 + gap * j:gap*timeSteps + gap * j:gap, 1]
                y_train[170000//gap * (i - 1) + j, :] = locals()['a' + str(i + 1)][gap*timeSteps + gap * j, whichMotion]
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    # for i in range(1, 40):
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
        layers = [64, 1]

        model.add(LSTM(
            layers[0],
            input_shape = (timeSteps, 1),
            return_sequences=False))
        # model.add(Dropout(0.5))

        # model.add(LSTM(
        #     layers[1],
        #     return_sequences=False))

        model.add(Dense(
            layers[1], activation = 'linear'))
        # model.add(Activation("linear"))
        start = time.time()
        model.compile(loss="mse", optimizer="adam")
        keras.optimizers.Adam(lr=0.0006)
        print("Compilation Time : ", time.time() - start)
        return model


    def run_network(model=None, epochs=0):
        if model is None:
            model = build_model()
        else:
            model = load_model(model)

        try:
            if epochs > 0:
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
        if epochs > 0:
            model.save('test49_'+str(whichMotion)+'.h5')
        return model, predicted


    # [mo, pred] = run_network(model='test49.h5', epochs=10)
    [mo, pred] = run_network(model=None, epochs = 50)

    # plt.figure()
    # plt.plot(a12[300:165300:3,4])
    # plt.plot(a12[300:165300:3,1])