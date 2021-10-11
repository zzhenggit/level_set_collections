# -*- coding: utf-8 -*-

import math
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
from scipy.ndimage import gaussian_filter
from scipy.ndimage import laplace

plt.ion()
fig1 = plt.figure(1)
fig2 = plt.figure(2)

def show_fig1(phi: np.ndarray):
    fig1.clf()
    ax1 = fig1.add_subplot(111, projection='3d')
    y, x = phi.shape
    x = np.arange(0, x, 1)
    y = np.arange(0, y, 1)
    X, Y = np.meshgrid(x, y)
    ax1.plot_surface(X, Y, -phi, rstride=2, cstride=2, color='r', linewidth=0, alpha=0.6, antialiased=True)
    ax1.contour(X, Y, phi, 0, colors='g', linewidths=2)


def show_fig2(phi: np.ndarray, img: np.ndarray):
    fig2.clf()
    contours = measure.find_contours(phi, 0)
    ax2 = fig2.add_subplot(111)
    ax2.imshow(img, interpolation='nearest', cmap=plt.get_cmap('gray'))
    for n, contour in enumerate(contours):
        ax2.plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')
    ax2.axis('off')

def draw_all(phi: np.ndarray, img: np.ndarray, pause=0.3):
    show_fig2(phi, img)
    show_fig1(phi)
    plt.pause(pause)

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
        compute the distance regularization term with the double-well potential p2
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