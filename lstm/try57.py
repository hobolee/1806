# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
import os
# from keras.utils.vis_utils import plot_model
from keras.callbacks import ReduceLROnPlateau
from keras import backend as K
from keras import regularizers
from keras import optimizers

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# time, wave.C1, surge, sway, heave, roll, pitch, yaw
fileNum = 40
for i in range(1, fileNum + 1):
    if i < 41:
        if i < 10:
            locals()['a' + str(i)] = np.loadtxt(r'D:\1806\data_aligned\C10' + str(i) + '.txt',
                                                skiprows=0)
        elif i < 17:
            locals()['a' + str(i)] = np.loadtxt(r'D:\1806\data_aligned\C1' + str(i) + '.txt',
                                                skiprows=0)
        elif i < 26:
            locals()['a' + str(i)] = np.loadtxt(
                r'D:\1806\data_aligned\C40' + str(i - 16) + '.txt', skiprows=0)
        else:
            locals()['a' + str(i)] = np.loadtxt(r'D:\1806\data_aligned\C4' + str(i - 16) + '.txt',
                                                skiprows=0)
gap = 20
timeSteps = 256
whichMotion = 2
trainNum = 39
pointNum = 167100

zero = np.zeros([gap*timeSteps, a1.shape[1]])
print(zero.shape)
print(a1.shape)
for i in range(1, fileNum + 1):
    locals()['a' + str(i)] = np.concatenate((zero, locals()['a' + str(i)]))
    print(locals()['a' + str(i)].shape)

# t_train = np.empty((trainNum * pointNum // gap, 1), dtype="float32")
X_train = np.empty((trainNum * pointNum // gap, timeSteps), dtype="float32")
y_train = np.empty((trainNum * pointNum // gap, 1), dtype="float32")
# t_test = np.empty(((fileNum - trainNum) * pointNum // gap, 1), dtype="float32")
X_test = np.empty(((fileNum - trainNum) * pointNum // gap, timeSteps), dtype="float32")
y_test = np.empty(((fileNum - trainNum) * pointNum // gap, 1), dtype="float32")
# t_train_plot = np.empty((trainNum * pointNum, 1), dtype="float32")
y_train_plot = np.empty((trainNum * pointNum, 1), dtype="float32")
# t_test_plot = np.empty(((fileNum - trainNum) * pointNum, 1), dtype="float32")
y_test_plot = np.empty(((fileNum - trainNum) * pointNum, 1), dtype="float32")
for i in range(1, fileNum + 1):
    for j in range(pointNum // gap):
        if i > trainNum:
            X_test[pointNum // gap * (i - trainNum - 1) + j, :] = locals()['a' + str(i)][
                                                                0 + gap * j:gap * timeSteps + gap * j:gap, 1]
            y_test[pointNum // gap * (i - trainNum - 1) + j] = locals()['a' + str(i)][
                gap * timeSteps + gap * j, whichMotion]
            # t_test[pointNum // gap * (i - trainNum - 1) + j] = locals()['a' + str(i)][
            #     gap * timeSteps + gap * j, 0]
        else:
            X_train[pointNum // gap * (i - 1) + j, :] = locals()['a' + str(i)][0 + gap * j:gap * timeSteps + gap * j:gap,
                                                      1]
            y_train[pointNum // gap * (i - 1) + j, :] = locals()['a' + str(i)][gap * timeSteps + gap * j, whichMotion]
            # t_train[pointNum // gap * (i - 1) + j, :] = locals()['a' + str(i)][gap * timeSteps + gap * j, 0]

for i in range(1, fileNum + 1):
    for j in range(pointNum):
        if i > trainNum:
            y_test_plot[pointNum * (i - trainNum - 1) + j] = locals()['a' + str(i)][gap * timeSteps + j, whichMotion]
            # t_test_plot[pointNum * (i - trainNum - 1) + j] = locals()['a' + str(i)][timeSteps + j, 0]
        else:
            y_train_plot[pointNum * (i - 1) + j, :] = locals()['a' + str(i)][gap * timeSteps + j, whichMotion]
            # t_train_plot[pointNum // gap * (i - 1) + j, :] = locals()['a' + str(i)][timeSteps + j, 0]

# for i in range(1, 40):
#    del locals()['a'+str(i)]
#    del locals()['b'+str(i)]
# X_train = X_train[170000//gap*17:170000//gap*28, :]
# y_train = y_train[170000//gap*17:170000//gap*28, :]
# X_test = X_train[-170000//gap:, :]
# y_test = y_train[-170000//gap:, :]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

t = np.arange(0, gap*(len(y_train)+len(y_test)))
t_plot = np.arange(0, gap*(len(y_train)+len(y_test)), gap)
print(t.shape)
print(t_plot.shape)

import time
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
import keras


def build_model():
    # class LossHistory(keras.callbacks.Callback):
    #     def on_train_begin(self, logs={}):
    #         self.losses = []
    #
    #     def on_batch_end(self, batch, logs={}):
    #         self.losses.append(logs.get('loss'))

    model = Sequential()
    # layers = [512, 1]
    layers = [512, 1]
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

    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9)
    model.compile(loss="mse", optimizer=sgd)
    # keras.optimizers.Adam(lr=0.001)
    # history = LossHistory()
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
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=4, mode='auto', factor=0.1)
            history = model.fit(
                X_train, y_train,
                # batch_size=8192, epochs=epochs, validation_split=0, shuffle=False)
                batch_size=2048, epochs=epochs, validation_split=0.1, callbacks=[reduce_lr], shuffle=True)
            trainLoss = history.history["loss"]
            valloss = history.history["val_loss"]
            np.savetxt("loss.txt", [trainLoss, valloss])

        predicted = model.predict(X_test)
        predicted = np.reshape(predicted, (predicted.size,))
        predicted_train = model.predict(X_train)
        predicted_train = np.reshape(predicted_train, (predicted_train.size,))
    except KeyboardInterrupt:
        return model, y_test, 0
    try:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(t_plot, np.concatenate((y_train[:, 0], y_test[:, 0])), label="test")
        plt.plot(t, np.concatenate((y_train_plot[:, 0], y_test_plot[:, 0])), label="no_gap")
        plt.plot(t_plot, np.concatenate((predicted_train, predicted[:])), label="pred")
        plt.legend(loc='upper left')
        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
    except Exception as e:
        print(str(e))
    if epochs > 0:
        model.save("test57_512_256.h5")
        print('model saved')
    return model, predicted


# [mo, pred] = run_network(model='test56_512_256.h5', epochs=20)
[mo, pred] = run_network(model=None, epochs = 10)


