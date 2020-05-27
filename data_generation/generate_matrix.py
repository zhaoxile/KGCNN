
import os
import matplotlib.image as img
import numpy as np
import cv2
import random
import h5py
import sys
import matplotlib.pyplot as plot
import tool_box.blur as blur
from setting import *





X  = np.zeros((SIZE_MATRIX,SIZE_KERNEL * SIZE_KERNEL))
for i in range(SIZE_MATRIX):
    r_lenth = 15 + 15 * random.random()
    r_lenth = int(r_lenth)
    angle = 45 + 90 * random.random()
    kernel = blur.motionblur(r_lenth, angle) #, -1, blur.gaussianblur([3, 3], np.random.random()*0.2 + 0.3))
    tmp = np.zeros((SIZE_KERNEL, SIZE_KERNEL))
    row, col = kernel.shape
    tmp[(SIZE_KERNEL - row) // 2: (SIZE_KERNEL - row) // 2 + row, (SIZE_KERNEL - col) // 2: (SIZE_KERNEL - col) // 2 + col] = kernel
    X[i, :]  = tmp.reshape(1,SIZE_KERNEL * SIZE_KERNEL)

X_mean = np.mean(X, axis= 0)
X_std = np.std(X, axis = 0)
X_std[X_std==0] = 1
for i in range(SIZE_KERNEL * SIZE_KERNEL):
    X[:,i] = (X[:, i] - X_mean[i]) / X_std[i]
U , sigma ,VT = np.linalg.svd(X)
V = VT.T
tsum = 0
Sig = sigma / sigma.sum()
for i in range(Sig.size):
    tsum = tsum + Sig[i]
    if tsum > PER_FOR_MATRIX:
        f = h5py.File('./matrix/matrix.h5', 'w')
        f['V'] = V[:, 0 : i + 1]
        f['t'] = i + 1
        f['mean'] = X_mean
        f['std'] = X_std
        f.close()
        print(i)
        break



