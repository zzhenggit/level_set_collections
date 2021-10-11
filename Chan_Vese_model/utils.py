# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import math
from skimage import measure

plt.ion()
fig1 = plt.figure(1)

def show_image(img, phi):
    fig1.clf()
    init_contours = measure.find_contours(phi, 0.5)
    show_image_and_segmentation(fig1, img, init_contours)
    fig1.suptitle("demo")
    plt.show()
    plt.pause(0.5)

def show_image_and_segmentation(fig, img, contours):
    ax2 = fig.add_subplot(111)
    ax2.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
    for n, contour in enumerate(contours):
        ax2.plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')
    ax2.axis('off')

def Heaviside (f, epsilon):
    h = 0.5 * (1 + (2 / math.pi) * np.arctan(f / epsilon))
    return h

def Dirac(f, epsilon):
    d = epsilon / (math.pi * (epsilon * epsilon + f * f))
    return d

def div(nx, ny):
    [_, nxx] = np.gradient(nx)
    [nyy, _] = np.gradient(ny)
    return nxx + nyy
