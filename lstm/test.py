# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 18:06:37 2018

@author: DELL
"""
#for i in range(1, 25):
#    if i < 13:
#        if i < 10:
#            locals()['a'+str(i)]=np.loadtxt(r'D:\1806\1806cal\C40'+str(i)+'-Full.dat', skiprows = 2)
#        else:
#            locals()['a'+str(i)]=np.loadtxt(r'D:\1806\1806cal\C4'+str(i)+'-Full.dat', skiprows = 2)
#    else:
#        if i < 22:
#            locals()['a'+str(i)]=np.loadtxt(r'D:\1806\DATA FILES\C40'+str(i-12)+'-Full.txt', skiprows = 2)
#        else:
#            locals()['a'+str(i)]=np.loadtxt(r'D:\1806\DATA FILES\C4'+str(i-12)+'-Full.txt', skiprows = 2)
plt.figure()
#plt.plot(a2[:,1])
#plt.plot(a13[:,1])
#plt.plot(a13[:,4])
#plt.plot(a13[:,-1])
X_test = np.empty((55000,100),dtype="float32")
y_test = np.empty((55000,1),dtype="float32")
for j in range(55000):
    X_test[j,:] = a13[0+j:300+j:3,-1]
    y_test[j] = a13[j,4] 
plt.plot(X_test[:,0])
plt.plot(y_test)