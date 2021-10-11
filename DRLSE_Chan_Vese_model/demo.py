# -*- coding: utf-8 -*-

'''
this code is a toy implementation to combine DRLSE and CV to form the region-based DRLSE.
The energy of which consists of the regularization term, edge term and area term.
Note: level set initialization and parameters are set empirically,
      which may need to be modified for different images.
'''

import numpy as np
from skimage.io import imread
from skimage.transform import resize
from evolution import evolution

img = imread('../images/plane2.bmp', True)
#img = imread('../images/plane4.jpg', True)

img = resize(img, (100, 100))
img = np.interp(img, [np.min(img), np.max(img)], [0, 255])

time_step = 0.1
iters = 100
mu = 0.003
# weight of edge term
lmda = 0.01
# weight of region term
alpha = 0.5
epsilon = 1.5
potential_function = 'double-well'

c0 = 10
initial_lsf = c0 * np.ones(img.shape)
# generate the initial region R0 as two rectangles
initial_lsf[9:55, 9:30] = -c0


if __name__ == '__main__':
    phi = evolution(initial_lsf, img, lmda, mu, alpha, epsilon, time_step, iters, potential_function, display = True)
