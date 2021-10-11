# -*- coding: utf-8 -*-

import numpy as np
import math
import matplotlib.pyplot as plt
from skimage import measure
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

def draw_all_2phase(phi: np.ndarray, img: np.ndarray, pause=0.3):
    show_fig2(phi, img)
    show_fig1(phi)
    plt.pause(pause)

def show_fig_multi1(phi: np.ndarray):
    fig1.clf()
    ax1 = fig1.add_subplot(111, projection='3d')
    y, x = phi[:,:,0].shape
    x = np.arange(0, x, 1)
    y = np.arange(0, y, 1)
    X, Y = np.meshgrid(x, y)
    ax1.plot_surface(X, Y, -phi[:,:,0], rstride=2, cstride=2, color='r', linewidth=0, alpha=0.6, antialiased=True)
    ax1.contour(X, Y, phi[:,:,0], 0, colors='g', linewidths=2)
    ax1.plot_surface(X, Y, -phi[:,:,1], rstride=2, cstride=2, color='yellow', linewidth=0, alpha=0.6, antialiased=True)
    ax1.contour(X, Y, phi[:,:,1], 0, colors='blue', linewidths=2)

def show_fig_multi2(phi: np.ndarray, img: np.ndarray):
    fig2.clf()
    contours1 = measure.find_contours(phi[:,:,0], 0)
    contours2 = measure.find_contours(phi[:,:,1], 0)
    ax2 = fig2.add_subplot(111)
    ax2.imshow(img, interpolation='nearest', cmap=plt.get_cmap('gray'))
    for n, contour in enumerate(contours1):
        ax2.plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')
    for n, contour in enumerate(contours2):
        ax2.plot(contour[:, 1], contour[:, 0], linewidth=2, color='blue')
    ax2.axis('off')

def draw_all_multiphase(phi: np.ndarray, img: np.ndarray, pause=0.3):
    show_fig_multi2(phi, img)
    show_fig_multi1(phi)
    plt.pause(pause)

def normalize01(f):
    fmin = np.min(f)
    fmax = np.max(f)
    f_ = (f-fmin) / (fmax-fmin)
    return f_

def Heaviside (f, epsilon):
    h = 0.5 * (1 + (2 / math.pi) * np.arctan(f / epsilon))
    return h

def Dirac(x, epsilon):
    f=(epsilon/math.pi)/(epsilon * epsilon + x * x)
    return f


def neumann_bound_cond(f):
    """
        Make a function satisfy Neumann boundary condition
    """
    g = f.copy()

    g[np.ix_([0, -1], [0, -1])] = g[np.ix_([2, -3], [2, -3])]
    g[np.ix_([0, -1]), 1:-1] = g[np.ix_([2, -3]), 1:-1]
    g[1:-1, np.ix_([0, -1])] = g[1:-1, np.ix_([2, -3])]
    return g

def curvature_central(phi) -> np.ndarray:
    [phi_y, phi_x] = np.gradient(phi)
    s = np.sqrt(np.square(phi_x) + np.square(phi_y))
    delta = 1e-10
    n_x = phi_x / (s + delta)  # add a small positive number to avoid division by zero
    n_y = phi_y / (s + delta)
    [_, nxx] = np.gradient(n_x)
    [nyy, _] = np.gradient(n_y)
    return nxx + nyy

def creat_gauss_kernel(kernel_size=3, sigma=1, k=1):
    if sigma == 0:
        sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8
    X = np.linspace(-k, k, kernel_size)
    Y = np.linspace(-k, k, kernel_size)
    x, y = np.meshgrid(X, Y)
    x0 = 0
    y0 = 0
    gauss = 1/(2*np.pi*sigma**2) * np.exp(- ((x -x0)**2 + (y - y0)**2)/ (2 * sigma**2))
    return gauss

def conv2(image, kernel, padding=0, strides=1):
    # Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break
    return output

def dist_reg_p2(phi):
    [phi_y, phi_x] = np.gradient(phi)
    s = np.sqrt(np.square(phi_x) + np.square(phi_y))
    a = (s >= 0) & (s <= 1)
    b = (s > 1)
    ps = a * np.sin(2 * np.pi * s) / (2 * np.pi) + b * (s - 1)
    dps = ((ps != 0) * ps + (ps == 0)) / ((s != 0) * s + (s == 0))
    return div(dps * phi_x - phi_x, dps * phi_y - phi_y) + laplace(phi, mode='nearest')

def div(nx: np.ndarray, ny: np.ndarray) -> np.ndarray:
    [_, nxx] = np.gradient(nx)
    [nyy, _] = np.gradient(ny)
    return nxx + nyy
