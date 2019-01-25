# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 15:55:54 2018

@author: DELL
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
a1=np.loadtxt(r'D:\VIM\90deg-u0.224-1-Model.dat', skiprows = 0)
a2=np.loadtxt(r'D:\VIM\90deg-u0.226-1-Model.dat', skiprows = 0)
a3=np.loadtxt(r'D:\VIM\90deg-u0.227-1-Model.dat', skiprows = 0)

plt.figure()
plt.plot(a2[:,2])





