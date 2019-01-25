# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 14:13:06 2018

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
a12=np.loadtxt(r'D:\1806\DATA FILES\C412-Full.txt')


#train = np.empty((550000,100),dtype="float32")
#label = np.empty((550000,1),dtype="float32")
X_train = np.empty((605000,100),dtype="float32")
y_train = np.empty((605000,1),dtype="float32")
X_test = np.empty((55000,100),dtype="float32")
y_test = np.empty((55000,1),dtype="float32")

for i in range(12):
    for j in range(55000):
        if i==11:
            X_test[j,:] = a12[0+j:300+j:3,1]
            y_test[j] = a12[300+j,2] 
        else:
            X_train[55000*i+j,:] = locals()['a'+str(i+1)][0+j:300+j:3,1]
            y_train[55000*i+j,:] = locals()['a'+str(i+1)][300+j,2]
#del a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

plt.plot(a1[:,1])
plt.plot(a1[:,2])
