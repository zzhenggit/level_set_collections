# -*- coding: utf-8 -*-

import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import laplace

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

def div(nx: np.ndarray, ny: np.ndarray) -> np.ndarray:
    [_, nxx] = np.gradient(nx)
    [nyy, _] = np.gradient(ny)
    return nxx + nyy

def Heaviside_1(x: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    f = (1/2) * (1 + x / sigma + (1 / math.pi) * (np.sin(math.pi * x / sigma)) )
    f [f > sigma] = 1
    f [f < -sigma] = 0
    return f

def Heaviside (f, epsilon):
    h = 0.5 * (1 + (2 / math.pi) * np.arctan(f / epsilon))
    return h

def dirac(x: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    f = (1 / 2 / sigma) * (1 + np.cos(np.pi * x / sigma))
    b = (x <= sigma) & (x >= -sigma)
    return f * b

def dist_reg_p2(phi):
    """
        compute the distance regularization term with the double-well potential p2 in equation (16)
    """
    [phi_y, phi_x] = np.gradient(phi)
    s = np.sqrt(np.square(phi_x) + np.square(phi_y))
    a = (s >= 0) & (s <= 1)
    b = (s > 1)
    ps = a * np.sin(2 * np.pi * s) / (2 * np.pi) + b * (s - 1)
    dps = ((ps != 0) * ps + (ps == 0)) / ((s != 0) * s + (s == 0))
    return div(dps * phi_x - phi_x, dps * phi_y - phi_y) + laplace(phi, mode='nearest')

def neumann_bound_cond(f):
    """
        Make a function satisfy Neumann boundary condition
    """
    g = f.copy()

    g[np.ix_([0, -1], [0, -1])] = g[np.ix_([2, -3], [2, -3])]
    g[np.ix_([0, -1]), 1:-1] = g[np.ix_([2, -3]), 1:-1]
    g[1:-1, np.ix_([0, -1])] = g[1:-1, np.ix_([2, -3])]
    return g

def get_g (img):
    img_smooth = gaussian_filter(img, sigma = 0.8)  # smooth image by Gaussian convolution
    [Iy, Ix] = np.gradient(img_smooth)
    f = np.square(Ix) + np.square(Iy)
    g = 1 / (1 + f)  # edge indicator function.
    return g
