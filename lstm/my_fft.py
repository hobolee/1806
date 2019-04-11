import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft

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
gap = 80
X_train = a1[:, 1]
y_train = a1[:, 3]
X_test = a2[:, 1]
y_test = a2[:, 3]
tt = a1[:, 0]
tt_gap = a1[0:-1:gap, 0]
yy_gap = a1[0:-1:gap, 3]
plt.figure()
plt.plot(tt, y_train,'r', tt_gap, yy_gap, 'b')


t = a1[:, 0]
Ts = 0.01*np.sqrt(56)
Fs = 1.0/Ts
n = len(y_train)
k = np.arange(n)
T = n/Fs
frq = k/T
frq = frq[range(int(n/2))]
yy = np.fft.fft(y_train)
y = yy/n
y = y[range(int(n/2))]

t1 = tt_gap
Ts1 = 0.01*np.sqrt(56)*gap
Fs1 = 1.0/Ts1
n1 = len(tt_gap)
k1 = np.arange(n1)
T1 = n1/Fs1
frq1 = k1/T1
frq1 = frq1[range(int(n1/2))]

yy1 = np.fft.fft(yy_gap)
y1 = yy1/n1
y1 = y1[range(int(n1/2))]

plt.figure()
plt.plot(frq*2*np.pi, abs(y), frq1*2*np.pi, abs(y1))
# plt.xlime(0, 1)
plt.show()



