# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 12:30:37 2018

@author: DELL
"""

import numpy as np
from matplotlib.pyplot import plot
from matplotlib.pyplot import show
import matplotlib.pyplot as plt
pred = np.load('test13-7.npy')
label = np.load('test13_label.npy')

N = 30
n = np.ones(N)
weights = n/N
sma = np.convolve(weights,pred[N-1:-N+1])
t = np.arange(N-1,len(pred))
#plot(t,pred[N-1:],lw=1)
plt.figure()
plot(t,label[N-1:],lw=2)
plot(t,sma,lw=3)



#N = 50
#n = np.ones(N)
#weights = n/N
#label = np.resize(label,(170000,))
#sma1 = np.convolve(weights,pred[N-1:-N+1])
#sma2 = np.convolve(weights,label[N-1:-N+1])
#t = np.arange(N-1,len(pred))
#plt.figure()
#plot(t,sma2,lw=2)
#plot(t,sma1,lw=3)
#
#N = 50
#n = np.ones(N)
#weights = n/N
#sma3 = np.convolve(weights,sma1[N-1:-N+1])
#sma4 = np.convolve(weights,sma2[N-1:-N+1])
#t = np.arange(N-1,len(sma1))
#plt.figure()
#plot(t,sma3,lw=2)
#plot(t,sma4,lw=3)
#
#N = 50
#n = np.ones(N)
#weights = n/N
#sma5 = np.convolve(weights,sma3[N-1:-N+1])
#sma6 = np.convolve(weights,sma4[N-1:-N+1])
#t = np.arange(N-1,len(sma3))
#plt.figure()
#plot(t,sma5,lw=2)
#plot(t,sma6,lw=3)
#
#N = 50
#n = np.ones(N)
#weights = n/N
#sma7 = np.convolve(weights,sma5[N-1:-N+1])
#sma8 = np.convolve(weights,sma6[N-1:-N+1])
#t = np.arange(N-1,len(sma5))
#plt.figure()
#plot(t,sma7,lw=2)
#plot(t,sma8,lw=3)
#
#N = 50
#n = np.ones(N)
#weights = n/N
#sma9 = np.convolve(weights,sma7[N-1:-N+1])
#sma10 = np.convolve(weights,sma8[N-1:-N+1])
#t = np.arange(N-1,len(sma7))
#plt.figure()
#plot(t,sma9,lw=2)
#plot(t,sma10,lw=3)
#
#N = 50
#n = np.ones(N)
#weights = n/N
#sma11 = np.convolve(weights,sma9[N-1:-N+1])
#sma12 = np.convolve(weights,sma10[N-1:-N+1])
#t = np.arange(N-1,len(sma9))
#plt.figure()
#plot(t,sma11,lw=2)
#plot(t,sma12,lw=3)
