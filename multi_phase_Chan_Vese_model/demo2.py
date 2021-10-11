# -*- coding: utf-8 -*-

'''
This code is the implementation of multi-phase level set in the paper
'A Multiphase Level Set Framework for Image Segmentation Using the Mumford and Shah Model'
In the code, two level sets for segmenting four phases are used.
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

mu = 0.0165 * 255 * 255
epsilon = 1
time_step = 0.1
iters = 300

phi = np.ones((img.shape[0], img.shape[1], 2))

for i in range (phi.shape[0]):
    for j in range (phi.shape[1]):
        phi[:, :, 0][i, j] = (-1) * np.sqrt(np.square(i - 45) + np.square(j - 30)) + 20
        phi[:, :, 1][i, j] = (-1) * np.sqrt(np.square(i - 45) + np.square(j - 70)) + 20

if __name__ == '__main__':
    phi = evolution(phi, img, mu, epsilon, time_step, iters, reinit = False, display = True)

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