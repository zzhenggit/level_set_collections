# -*- coding: utf-8 -*-

'''
This code is the implementation of the two-phase level set for the following paper:
A Level Set Method for Image Segmentation in the Presence of Intensity Inhomogeneities with Application to MRI ",
IEEE Trans. Image Processing, vol. 20 (7), pp. 2007-2016, 2011.
'''

from skimage.io import imread
from scipy.ndimage import gaussian_filter
from lse_bfe import lse_bfe
from utils import *

Img = imread('../images/heart_ct.bmp', True)
Img = np.array(Img, dtype='float32')

A = 255
Img = A * normalize01(Img)  # rescale the image intensities
nu = 0.001 * A * A  # coefficient of arc length term
sigma = 4  # scale parameter that specifies the size of the neighborhood
iter_outer = 50
iter_inner = 10  # inner iteration for level set evolution
timestep = 0.1
mu = 1  # coefficient for distance regularization term (regularize the level set function)
c0 = 1

# % initialize level set function
initialLSF = c0 * np.ones_like(Img)
initialLSF[30:90, 50:90] = -c0
u = initialLSF

epsilon = 1
b = np.ones_like(Img)  # initialize bias field

KI = gaussian_filter(Img, sigma=sigma, truncate=2)
KONE = gaussian_filter(np.ones_like(Img), sigma=sigma, truncate=2)

if __name__ == '__main__':
    for i in range(iter_outer):
        u, b, C = lse_bfe(u, Img, b, sigma, KONE, nu, timestep, mu, epsilon, iter_inner)

        if (i % 2) == 0:
            draw_all_2phase(u, Img)

    Mask = Img > 10
    Img_corrected = normalize01(Mask * Img / (b + (b == 0))) * 255

    fig3 = plt.figure(3)
    fig3.suptitle("original image")
    ax3 = fig3.add_subplot(111)
    ax3.imshow(Img, cmap=plt.get_cmap('gray'))
    ax3.axis('off')
    plt.pause(5)

    fig4 = plt.figure(4)
    fig4.suptitle("Bias corrected image")
    ax4 = fig4.add_subplot(111)
    ax4.imshow(Img_corrected, cmap=plt.get_cmap('gray'))
    ax4.axis('off')
    plt.pause(5)

    fig5 = plt.figure(5)
    fig5.suptitle("Bias field")
    ax5 = fig5.add_subplot(111)
    ax5.imshow(normalize01(b) * 255, cmap=plt.get_cmap('gray'))
    ax5.axis('off')
    plt.pause(5)

    plt.show()