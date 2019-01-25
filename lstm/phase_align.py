# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 14:48:43 2018

@author: DELL
"""

import numpy as np
import matplotlib.pyplot as plt
#for i in range(1, 81):
#    if i < 41:
#        if i < 10:
#            locals()['a'+str(i)]=np.loadtxt(r'D:\1806\1806cal\C10'+str(i)+'-Full.dat', skiprows = 2)
#        elif i < 17:
#            locals()['a'+str(i)]=np.loadtxt(r'D:\1806\1806cal\C1'+str(i)+'-Full.dat', skiprows = 2)
#        elif i < 26:
#            locals()['a'+str(i)]=np.loadtxt(r'D:\1806\1806cal\C40'+str(i-16)+'-Full.dat', skiprows = 2)
#        else:
#            locals()['a'+str(i)]=np.loadtxt(r'D:\1806\1806cal\C4'+str(i-16)+'-Full.dat', skiprows = 2)
#    else:
#        if i < 50:
#            locals()['b'+str(i-40)]=np.loadtxt(r'D:\1806\DATA FILES\C10'+str(i-40)+'-Full.txt', skiprows = 2)
#        elif i < 57:
#            locals()['b'+str(i-40)]=np.loadtxt(r'D:\1806\DATA FILES\C1'+str(i-40)+'-Full.txt', skiprows = 2)
#        elif i < 66:
#            locals()['b'+str(i-40)]=np.loadtxt(r'D:\1806\DATA FILES\C40'+str(i-56)+'-Full.txt', skiprows = 2)
#        else:
#            locals()['b'+str(i-40)]=np.loadtxt(r'D:\1806\DATA FILES\C4'+str(i-56)+'-Full.txt', skiprows = 2)
#            
a18 = np.loadtxt(r'D:\1806\1806cal\C402-Full.dat', skiprows = 2)
b18 = np.loadtxt(r'D:\1806\DATA FILES\C402-Full.txt', skiprows = 2)
C402 = np.correlate(a18[:,1],b18[:,1])
plt.figure()
plt.plot(a18[:,1])
plt.plot(b18[:,1])
lag = np.where(C402==np.max(C402))
plt.figure()
plt.plot(a18[lag[0][0]:,1])
plt.plot(b18[:,1])