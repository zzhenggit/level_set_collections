# -*- coding: utf-8 -*-

'''
This code is the implementation of two-phase level set for the following paper:
T. F. Chan and L. A. Vese, "Active contours without edges,"
in IEEE Transactions on Image Processing, vol. 10, no. 2, pp. 266-277, Feb. 2001, doi: 10.1109/83.902291.
Note: level set initialization and parameters are set empirically,
      which may need to be modified for different images.
'''

import numpy as np
from skimage.io import imread
from skimage.transform import resize
from evolution import  evolution


img = imread('../images/fin1.bmp', as_gray=True)
img = resize(img, (100, 100))
img = np.interp(img, [np.min(img), np.max(img)], [0, 255])

phi= np.zeros_like(img)

for i in range (phi.shape[0]):
    for j in range (phi.shape[1]):
        phi[i,j] = (-1) * np.sqrt(np.square(i - 50) + np.square(j-50)) + 40

lambda_1 = 1
lambda_2 = 1
mu = 0.2 * 255 * 255
epsilon = 1
time_step = 0.1
iters = 100

if __name__ == '__main__':
    phi = evolution(phi, img, lambda_1, lambda_2, mu, epsilon, time_step, iters,
                    reinit = False, display= True)