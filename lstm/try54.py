# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
import os
# from keras.utils.vis_utils import plot_model
from keras.callbacks import ReduceLROnPlateau
from keras import backend as K

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# time, wave.C1, surge, sway, heave, roll, pitch, yaw
fileNum = 2
for i in range(1, fileNum + 1):
    if i < 41:
        if i < 10:
            locals()['a' + str(i)] = np.loadtxt(r'D:\1806\data_aligned\C30' + str(i) + '.txt',
                                                skiprows=0)
        elif i < 17:
            locals()['a' + str(i)] = np.loadtxt(r'D:\1806\data_aligned\C3' + str(i) + '.txt',
                                                skiprows=0)
        elif i < 26:
            locals()['a' + str(i)] = np.loadtxt(
                r'D:\1806\data_aligned\C50' + str(i - 16) + '.txt', skiprows=0)
        else:
            locals()['a' + str(i)] = np.loadtxt(r'D:\1806\data_aligned\C5' + str(i - 16) + '.txt',
                                                skiprows=0)
gap = 2
timeSteps = 75
whichMotion = 3
trainNum = 1
X_train = np.empty((trainNum * 170000 // gap, timeSteps), dtype="float32")
y_train = np.empty((trainNum * 170000 // gap, 1), dtype="float32")
X_test = np.empty(((fileNum - trainNum) * 170000 // gap, timeSteps), dtype="float32")
y_test = np.empty(((fileNum - trainNum) * 170000 // gap, 1), dtype="float32")
for i in range(1, fileNum + 1):
    for j in range(170000 // gap):
        if i > trainNum:
            X_test[170000 // gap * (i - trainNum - 1) + j, :] = locals()['a' + str(i)][
                                                                0 + gap * j:gap * timeSteps + gap * j:gap, 1]
            y_test[170000 // gap * (i - trainNum - 1) + j] = locals()['a' + str(i)][
                gap * timeSteps + gap * j, whichMotion]
        else:
            X_train[170000 // gap * (i - 1) + j, :] = locals()['a' + str(i)][0 + gap * j:gap * timeSteps + gap * j:gap,
                                                      1]
            y_train[170000 // gap * (i - 1) + j, :] = locals()['a' + str(i)][gap * timeSteps + gap * j, whichMotion]
# y_train = y_train
# y_test = y_test

# for i in range(1, 40):
#    del locals()['a'+str(i)]
#    del locals()['b'+str(i)]

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
import time
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
import keras


def build_model():
    model = Sequential()
    layers = [128, 1]

    model.add(LSTM(
        layers[0],
        input_shape=(timeSteps, 1),
        return_sequences=False))
    # model.add(Dropout(0.5))

    model.add(Dense(
        layers[1], activation='linear'))
    # model.add(Activation("linear"))
    start = time.time()

    def my_loss(y_true, y_pred):
        # return np.array(1)
        return K.mean(K.abs(y_pred - y_true), axis=-1)

    model.compile(loss="mse", optimizer="adam")
    keras.optimizers.Adam(lr=0.0000001)
    print("Compilation Time : ", time.time() - start)
    return model


def run_network(model=None, epochs=0):
    if model is None:
        model = build_model()
    else:
        model = load_model(model)
    # plot_model(model, to_file='model2.png', show_shapes=True)

    try:
        if epochs > 0:
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=4, mode='auto')
            model.fit(
                X_train, y_train,
                # batch_size=8192, epochs=epochs, validation_split=0, shuffle=False)
                batch_size=2048, epochs=epochs, validation_split=0.1, callbacks=[reduce_lr])
        predicted = model.predict(X_test)
        predicted = np.reshape(predicted, (predicted.size,))
        predicted_train = model.predict(X_train)
        predicted_train = np.reshape(predicted_train, (predicted_train.size,))
    except KeyboardInterrupt:
        return model, y_test, 0
    try:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.concatenate((y_train[:, 0], y_test[:, 0])), label="test")
        plt.plot(np.concatenate((predicted_train, predicted[:])), label="pred")
        plt.legend(loc='upper left')
        plt.show()
    except Exception as e:
        print(str(e))
    if epochs > 0:
        model.save("test54.h5")
        print('model saved')
    return model, predicted


# [mo, pred] = run_network(model='test54.h5', epochs=50)
[mo, pred] = run_network(model=None, epochs = 30)

# plt.figure()
# plt.plot(a12[300:165300:3,4])
# plt.plot(a12[300:165300:3,1])
