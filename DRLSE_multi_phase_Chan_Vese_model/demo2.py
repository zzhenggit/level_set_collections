# -*- coding: utf-8 -*-

'''
This code is a toy implementation to combine the multi-phase level set and DRLSE
to form a multi-phase region-based DRLSE.

In the code, two level sets for segmenting four phases are used.
The total energy terms consists of regularization term, area term and edge term.
Note: level set initialization and parameters are set empirically,
      which may need to be modified for different images.
'''

from skimage.io import imread
from skimage.transform import resize
from evolution2 import evolution
from utils import *

img = imread('../images/fourphase1.png', True)
img = resize(img, (100, 100))
img = np.interp(img, [np.min(img), np.max(img)], [0, 255])

time_step = 0.5
iters = 100
mu = 0.003
# edge term
lmda = 0.01
# region term
alpha = 2
epsilon = 1.5
potential_function = 'double-well'

phi = np.ones((img.shape[0], img.shape[1], 2)) * 10

phi[10:50, 10:50, 0] = -10
phi[60:90, 60:90, 1] = -10

if __name__ == '__main__':
    phi = evolution(phi, img, mu, epsilon, potential_function, lmda, alpha, time_step, iters, display = True)

    H1 = phi[:,:,0] > 0
    H2 = phi[:,:,1] > 0

    member_function = np.zeros((img.shape[0], img.shape[1], 4))
    member_function[:, :, 0] = H1 * H2
    member_function[:, :, 1] = H1 * (1-H2)
    member_function[:, :, 2] = (1-H1) * H2
    member_function[:, :, 3] = (1-H1) * (1-H2)

    # show each segmented phase
    for i in range (4):
        fig = plt.figure(2+i)
        ax2 = fig.add_subplot(111)
        ax2.imshow(member_function[:, :, i] * 255)
        ax2.axis('off')
        plt.pause(3)
        plt.show()