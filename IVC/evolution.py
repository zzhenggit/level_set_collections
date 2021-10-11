# -*- coding: utf-8 -*-

import numpy as np
from scipy.ndimage import gaussian_filter
from utils import *

def evolution(phi_0, img, mu, sigma, iters, time_step, display = True):
    phi = phi_0.copy()

    for i in range (iters):
        [phi_y, phi_x] = np.gradient(phi)

        # we use the standard Heaviside function which yields similar results to regularized one.
        c1 = (np.sum (img * (phi < 0))) / (np.sum (phi < 0))
        c2 = (np.sum (img * (phi >= 0))) / (np.sum (phi >= 0))

        spf = img - (c1 + c2) / 2
        spf = spf / (np.max(abs(spf)))

        phi = phi + time_step * (mu * spf * np.sqrt(np.square(phi_y) + np.square(phi_x)))

        if (i % 10 == 0 and display):
            show_image(img, phi)

        # unlike the original Matlab code, the height of phi is set to 10 here
        # since the Gaussian filter tends to make (phi < 0) vanish if |phi| is small
        phi = 10 * (np.asarray((phi >= 0),dtype='int8') - np.asarray((phi < 0), dtype= 'int8')) # the selective step
        phi = gaussian_filter(phi, sigma)

    return phi