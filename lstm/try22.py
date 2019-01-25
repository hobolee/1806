# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 15:06:23 2018

@author: DELL
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras.models import load_model
from sklearn.metrics import mean_squared_error
import os
from pandas import datetime
from pandas import DataFrame
from pandas import concat
from pandas import Series
from sklearn.preprocessing import MinMaxScaler


os.environ['CUDA_VISIBLE_DEVICES'] = "0"
a1=np.loadtxt(r'D:\1806\DATA FILES\C111-Full.txt')
data1 = Series(a1[12000:12150,3])
plt.plot(data1)
data2 = data1.values
#data1 = Series([1,3,3,4,5,7,7,8,9,12])
#data2 = data1.values
# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df

# transform to supervised learning
#supervised = timeseries_to_supervised(data2, 1)
#print(supervised.head())

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# transform scale
def scale(train, test):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled
# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = np.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM

def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
        print(i)
    return model

# make a one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]


# transform data to be stationary
diff_values = difference(data2, 1)
# transform data to be supervised learning
supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values

# split data into train and test-sets
train, test = supervised_values[0:-50], supervised_values[-50:]

# transform the scale of the data
#scaler, train_scaled, test_scaled = scale(train, test)
train_scaled, test_scaled = train, test

# fit the model
lstm_model = fit_lstm(train_scaled, 1, 100, 5)
# forecast the entire training dataset to build up state for forecasting
#train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
#lstm_model.predict(train_reshaped, batch_size=1)



#尝试只输入一个值
#input_x = list()
#input_x.append(test_scaled[0, 0:-1])
#input_x.append(test_scaled[0, 0:-1])
history = data2[len(train)]
predictions = list()
for i in range(len(test_scaled)):
    # inpmake one-step forecast
    if i == 0:
        X = test_scaled[i, 0:-1]
    else:
        X = np.resize(yhat,(1))
    yhat = forecast_lstm(lstm_model, 1, X)
    # invert scaling
#    yhat = invert_scale(scaler, X, yhat)
    # invert differencing
#    yhat = inverse_difference(history, yhat, -1)
    yhat = history+yhat
    history = yhat
    # store forecast
    predictions.append(yhat)
    expected = data2[len(train) + i +1]
    print('Time=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))

# report performance
#rmse = np.sqrt(mean_squared_error(data3, predictions))
#print('Test RMSE: %.3f' % rmse)
#lstm_model.save('test20.h5')
# line plot of observed vs predicted
plt.figure()
plt.plot(data2[len(train)+1:])
plt.plot(predictions)
plt.show()




#原程序
# walk-forward validation on the test data
#predictions = list()
#for i in range(len(test_scaled)):
#	# make one-step forecast
#	X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
#	yhat = forecast_lstm(lstm_model, 1, X)
#	# invert scaling
#	yhat = invert_scale(scaler, X, yhat)
#	# invert differencing
#	yhat = inverse_difference(data2, yhat, len(test_scaled)+1-i)
#	# store forecast
#	predictions.append(yhat)
#	expected = data2[len(train) + i + 1]
#	print('Time=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))
#
## report performance
#rmse = np.sqrt(mean_squared_error(data2[-20:], predictions))
#print('Test RMSE: %.3f' % rmse)
## line plot of observed vs predicted
#plt.plot(data2[-20:])
#plt.plot(predictions)
#plt.show()

##用预测值代替输入值
#predictions = list()
#for i in range(len(test_scaled)):
#    # make one-step forecast
#    X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
#    yhat = forecast_lstm(lstm_model, 1, X)
#    # invert scaling
#    yhat = invert_scale(scaler, X, yhat)
#    # invert differencing
#    yhat = inverse_difference(data2, yhat, len(test_scaled)+1-i)
#    if i<19:
#        test_scaled[i+1, 0:-1] = yhat   
#    # store forecast
#    predictions.append(yhat)
#    expected = data2[len(train) + i + 1]
#    print('Time=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))
#
## report performance
#rmse = np.sqrt(mean_squared_error(data2[-100:], predictions))
#print('Test RMSE: %.3f' % rmse)
## line plot of observed vs predicted
#plt.plot(data2[-100:])
#plt.plot(predictions)
#plt.show()


#history = data2[len(train)]
#
#i=1
#if i == 0:
#    X = test_scaled[i, 0:-1]
#else:
#    X = np.resize(yhat,(1))
#yhat = forecast_lstm(lstm_model, 1, X)
## invert scaling
#yhat = invert_scale(scaler, X, yhat)
## invert differencing
##yhat = inverse_difference(history, yhat, -1)
#yhat = history+yhat
#history = yhat
## store forecast
#predictions.append(yhat)
#expected = data2[len(train) + i + 1 ]
#print('Time=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))