# -*- coding: utf-8 -*-

'''
This code is the implementation of the three-phase level set for the following paper:
A Level Set Method for Image Segmentation in the Presence of Intensity Inhomogeneities with Application to MRI ",
IEEE Trans. Image Processing, vol. 20 (7), pp. 2007-2016, 2011.
'''

from skimage.io import imread
from scipy.ndimage import gaussian_filter
from lse_bfe_3Phase import lse_bfe_3Phase
from utils import *

Img = imread('../images/myBrain_axial.bmp', True)
Img = np.array(Img, dtype='float32')

A = 255
Img = A * normalize01(Img)  # rescale the image intensities
Mask = Img > 5
nu = 0.001 * A * A  # coefficient of arc length term
sigma = 4  # scale parameter that specifies the size of the neighborhood
iter_outer = 50
iter_inner = 10  # inner iteration for level set evolution
timestep = 0.1
mu = 0.1 / timestep  # coefficient for distance regularization term (regularize the level set function)
c0 = 5
epsilon = 1

# initialization of bias field and level set function
b = np.ones_like(Img)
initialLSF = np.random.randn(Img.shape[0],Img.shape[1],2)
initialLSF[:,:,1] = Mask

u = np.sign(initialLSF)

KONE = gaussian_filter(np.ones_like(Img), sigma=sigma, truncate=2)

C = None

if __name__ == '__main__':
    for i in range(iter_outer):
        u, b, C = lse_bfe_3Phase(u, Img, b, sigma, KONE, nu, timestep, mu, epsilon, iter_inner)

        if (i % 1) == 0:
            draw_all_multiphase(u, Img)

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
    ax5.imshow(Mask * normalize01(b) * 200, cmap=plt.get_cmap('gray'))
    ax5.axis('off')
    plt.pause(5)

    H1 =  u[:,:,0] > 0
    H2 =  u[:,:,1] > 0
    M1=H1 * H2
    M2=H1 *(1-H2)
    M3=(1-H1)

    fig6 = plt.figure(6)
    Img_seg=C[0] *M1+C[1]*M2+C[2]*M3 # three regions are labeled with C1, C2, C3
    fig6.suptitle("Segmented Image")
    ax6 = fig6.add_subplot(111)
    ax6.imshow(Img_seg, cmap=plt.get_cmap('gray'))
    ax6.axis('off')
    plt.pause(5)

    plt.show()