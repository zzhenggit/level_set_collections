# -*- coding: utf-8 -*-

'''
This code is the implementation of the level set for the following paper:
"Active contours with selective local or global segmentation: a new variational approach and level set method"
in Image and Vision Computing, 2010.
'''

import numpy as np
from skimage.io import imread
from evolution import evolution

#img = imread('../images/galaxy.jpg', True)
img = imread('../images/plane4.jpg', True)
img = np.array(img, dtype='float32')

# unlike the original Matlab code, the height of phi is set to 10 here
# since the Gaussian filter tends to make (phi < 0) vanish if |phi| is small
phi = np.ones(img.shape) * 10
phi[10:img.shape[0] - 10, 10:img.shape[1] - 10] = -10
phi_0 = - phi

sigma = 1
delt = 1
Iter = 200
mu = 20  # this parameter needs to be tuned according to the images

if __name__ == '__main__':
    phi = evolution(phi_0, img, mu, sigma, Iter, delt, display = True)
