# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 18:51:01 2018

@author: DELL
"""


import numpy as np
from pandas import Series
import matplotlib.pyplot as plt
pred = np.load('test39.npy')
label = np.load('test39_label.npy')
pred = np.resize(pred, (len(pred),))
label = np.resize(label, (len(label),))
N =20
n = np.ones(N)
weights = n/N
sma = np.convolve(weights,pred[N-1:-N+1])
y = np.convolve(weights,label[N-1:-N+1])
t = np.arange(N-1,len(pred))
#plt.plot(t,pred[N-1:],lw=1)
plt.figure()
plt.plot(t,y,lw=1)
plt.plot(t,sma*1.5,lw=2)

def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)
p = difference(sma)
l = difference(y)

#plt.figure()
#plt.plot(p,lw=1)
#plt.plot(l,lw=2)


N =50
n = np.ones(N)
weights = n/N
pp = np.convolve(weights,p[N-1:-N+1])
ll = np.convolve(weights,l[N-1:-N+1])
t = np.arange(N-1,len(t))[0:-1]
#plot(t,pred[N-1:],lw=1)
plt.figure()
a1,=plt.plot(t, ll*0.6, label='prediction of W3R6')
a2,=plt.plot(t, pp, label='surge of W3R6')
plt.legend(loc='upper left')


ppp = difference(pp)
lll = difference(ll)



N =50
n = np.ones(N)
weights = n/N
pppp = np.convolve(weights,ppp[N-1:-N+1])
llll = np.convolve(weights,lll[N-1:-N+1])
#plot(t,pred[N-1:],lw=1)
#plt.figure()
#plt.plot(pppp)
#plt.plot(llll)
