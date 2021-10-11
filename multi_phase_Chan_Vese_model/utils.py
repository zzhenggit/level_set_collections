# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import math

def show_image_and_segmentation_all_two(fig, img, contours_1, contours_2, seeds=None):
    ax2 = fig.add_subplot(111)
    ax2.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
    for n, contour in enumerate(contours_1):
        ax2.plot(contour[:, 1], contour[:, 0], linewidth=1, color='red')
    for n, contour in enumerate(contours_2):
        ax2.plot(contour[:, 1], contour[:, 0], linewidth=1, color='green')
    if (seeds is not None):
        h_idx, w_idx = np.where(seeds[0] > 0)
        ax2.plot(w_idx, h_idx, linewidth=1, color='red')
        h_idx, w_idx = np.where(seeds[1] > 0)
        ax2.plot(w_idx, h_idx, linewidth=1, color='blue')
    ax2.axis('off')

def show_image_and_segmentation_all_three(fig, img, contours_1, contours_2, contours_3, seeds=None):
    ax2 = fig.add_subplot(111)
    ax2.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
    for n, contour in enumerate(contours_1):
        ax2.plot(contour[:, 1], contour[:, 0], linewidth=1, color='red')
    for n, contour in enumerate(contours_2):
        ax2.plot(contour[:, 1], contour[:, 0], linewidth=1, color='green')
    for n, contour in enumerate(contours_3):
        ax2.plot(contour[:, 1], contour[:, 0], linewidth=1, color= 'blue')
    if (seeds is not None):
        h_idx, w_idx = np.where(seeds[0] > 0)
        ax2.plot(w_idx, h_idx, linewidth=1, color='red')
        h_idx, w_idx = np.where(seeds[1] > 0)
        ax2.plot(w_idx, h_idx, linewidth=1, color='blue')
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
