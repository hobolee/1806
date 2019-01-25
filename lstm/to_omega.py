# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 09:31:01 2018

@author: DELL
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft
import seaborn
pred = np.load('test45_yaw.npy')
label = np.load('test45_yaw_label.npy')
pred = np.resize(pred, (len(pred),))
label = np.resize(label, (len(label),))
t = np.linspace(0,3180.40878,42500)
#N =5
#n = np.ones(N)
#weights = n/N
#sma = np.convolve(weights,pred[N-1:-N+1])
#y = np.convolve(weights,label[N-1:-N+1])
#plt.figure()
#plt.plot(sma)
#plt.plot(y)
sma = pred
y = label


sampling_rate = 5
fft_size = len(y)
xf1 = np.fft.rfft(y)/fft_size
freqs1 = np.linspace(0, sampling_rate/2, fft_size/2+1)
xf2 = np.fft.rfft(sma)/fft_size
freqs2 = np.linspace(0, sampling_rate/2, fft_size/2+1)
plt.figure()
plt.plot(freqs1, xf1)
plt.plot(freqs2, xf2)

#x = np.linspace(0,1,100)
#yy = fft(label)
#yreal = yy.real
#yimag = yy.imag
#yf = abs(fft(label))
#yf1 = abs(fft(label))/len(x)
#yf2 = yf1[range(int(len(x)/2))]
#
#xf = np.arange(len(label))
#xf1 = xf
#xf2 = xf[range(50)]
#
#plt.subplot(221)
#plt.plot(t,label)
#plt.title('Original wave')
#plt.subplot(222)
#plt.plot(xf,yf,'r')
#plt.title('FFT of Mixed wave(two side frequency range)')
#plt.subplot(223)
#plt.plot(xf1,yf1,'g')
#plt.title('FFT of Mixed wave(normalization)')
#plt.subplot(224)
#plt.plot(xf2,yf2,'b')
#plt.title('FFT of Mixed wave')
#
#y = np.sin(2*np.pi*25*t)


