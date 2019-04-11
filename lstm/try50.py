# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 22:14:59 2018

@author: DELL
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
#time, wave.C1, surge, sway, heave, roll, pitch, yaw
for i in range(1, 2):
    if i < 41:
        if i < 10:
            locals()['a' + str(i)] = np.loadtxt(r'C:\Users\29860\Documents\1806\data_aligned\C30' + str(i) + '.txt',
                                                skiprows=0)
        elif i < 17:
            locals()['a' + str(i)] = np.loadtxt(r'C:\Users\29860\Documents\1806\data_aligned\C3' + str(i) + '.txt',
                                                skiprows=0)
        elif i < 26:
            locals()['a' + str(i)] = np.loadtxt(
                r'C:\Users\29860\Documents\1806\data_aligned\C50' + str(i - 16) + '.txt', skiprows=0)
        else:
            locals()['a' + str(i)] = np.loadtxt(r'C:\Users\29860\Documents\1806\data_aligned\C5' + str(i - 16) + '.txt',
                                                skiprows=0)
gap = 1
timeSteps = 75*8
whichMotion = 3
trainNum = 1
data = np.concatenate((a1[:170000:gap, 1, np.newaxis], a1[:170000:gap, whichMotion, np.newaxis]-10), axis=1)
for i in range(2, 2):
    data = np.concatenate((data, np.concatenate((locals()['a'+str(i)][:170000:gap, 1, np.newaxis],
                                                 locals()['a'+str(i)][:170000:gap, whichMotion, np.newaxis]), axis=1)))
print('data', data.shape)
# X_train = np.empty((trainNum*170000//gap, timeSteps), dtype="float32")
# y_train = np.empty((trainNum*170000//gap, 1), dtype="float32")
# X_test = np.empty(((40-trainNum)*170000//gap, timeSteps), dtype="float32")
# y_test = np.empty(((40-trainNum)*170000//gap, 1), dtype="float32")
# for i in range(1, 41):
#     for j in range(170000//gap):
#         if i > trainNum:
#             X_test[170000//gap * (i - trainNum - 1) + j, :] = locals()['a' + str(i)][0 + gap * j:gap*timeSteps + gap * j:gap, 1]
#             y_test[170000//gap * (i - trainNum - 1) + j] = locals()['a' + str(i)][gap*timeSteps + gap * j, whichMotion]
#         else:
#             X_train[170000//gap * (i - 1) + j, :] = locals()['a' + str(i)][0 + gap * j:gap*timeSteps + gap * j:gap, 1]
#             y_train[170000//gap * (i - 1) + j, :] = locals()['a' + str(i)][gap*timeSteps + gap * j, whichMotion]
def get_train_data(batch_size=4096,time_step=timeSteps,train_begin=0,train_end=trainNum*170000//gap):
    batch_index=[]
    data_train=data[train_begin:train_end]
    mean=np.mean(data_train,axis=0)
    std=np.std(data_train,axis=0)
    normalized_train_data=(data_train-mean)/std  #标准化
    train_x,train_y=[],[]   #训练集
    for i in range(len(normalized_train_data)-time_step):
       if i % batch_size==0:
           batch_index.append(i)
       x=normalized_train_data[i:i+time_step,0,np.newaxis]
       y=normalized_train_data[i+time_step-1,1]
       train_x.append(x.tolist())
       train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data)-time_step))
    # print('X_train1', np.shape(train_x))
    # print('y_train1', np.shape(train_y))
    return batch_index,np.array(train_x),np.array(train_y),mean,std

def get_test_data(time_step=timeSteps,test_begin=0):
    data_test=data[test_begin:]
    mean=np.mean(data_test,axis=0)
    std=np.std(data_test,axis=0)
    normalized_test_data=(data_test-mean)/std  #标准化
    size=(len(normalized_test_data)+time_step-1)//time_step  #有size个sample
    test_x,test_y=[],[]
    for i in range(size-1):
       x=normalized_test_data[i*time_step:(i+1)*time_step,0,np.newaxis]
       y=normalized_test_data[i*time_step:(i+1)*time_step,1]
       test_x.append(x.tolist())
       test_y.extend(y)
    test_x.append((normalized_test_data[(i+1)*time_step:,0]).tolist())
    test_y.extend((normalized_test_data[(i+1)*time_step:,1]).tolist())
    # print('X_test', np.shape(test_x))
    return mean,std,np.array(test_x),np.array(test_y)

batch_index, X_train, y_train, mean_train, std_train=get_train_data()
batch_index_test, X_test, y_test, mean_test, std_test = get_train_data()
print('X_train', X_train.shape)
print('X_test', X_test.shape)
print('y_train', y_train.shape)
print('y_test', y_test.shape)

# X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

import time
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
import keras


def build_model():
    model = Sequential()
    layers = [32, 1]

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
                # batch_size=8192, epochs=epochs, validation_split=0, shuffle=False)
                batch_size=8192, epochs=epochs, validation_split=0.1)
        predicted = model.predict(X_test)
        predicted = predicted*std_train[1]+mean_train[1]
        predicted = np.reshape(predicted, (predicted.size,))
        predicted_train = model.predict(X_train)
        predicted_train = predicted_train*std_train[1]+mean_train[1]
        predicted_train = np.reshape(predicted_train, (predicted_train.size,))
    except KeyboardInterrupt:
        return model, y_test, 0
    try:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.concatenate((y_train*std_train[1]+mean_train[1]+10, y_test*std_train[1]+mean_train[1]+10)), label="test")
        plt.plot(np.concatenate((predicted_train+10, predicted+10)), label="pred")
        plt.legend(loc='upper left')
        plt.show()
    except Exception as e:
        print(str(e))
    if epochs > 0:
        model.save('test50.h5')
    return model, predicted


# [mo, pred] = run_network(model='test50.h5', epochs=10)
[mo, pred] = run_network(model=None, epochs = 20)

# plt.figure()
# plt.plot(a12[300:165300:3,4])
# plt.plot(a12[300:165300:3,1])