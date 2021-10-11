# -*- coding: utf-8 -*-

from utils import *

def evolution(phi_0, I, lambda_1, lambda_2, mu, epsilon, time_step, iters, reinit = False, display = True):
    """
    :param phi_0: initial level set function
    :param I: the input image to be segmented
    :param lambda_1: weight of inside C
    :param lambda_2: weight of outside C
    :param mu: weight of length term
    :param epsilon: width of Dirac Delta function
    :param time_step: time step
    :param iters: number of iterations
    :param reinit: whether reinitialize the level set
    :param display: whether display the evolution
    """
    phi = phi_0.copy()

    for k in range(iters):
        # display evolution
        if (display):
            show_image(I, phi)

        phi = Chan_Vese_model_update(phi, I, lambda_1, lambda_2, mu, epsilon, time_step)

        # optional reinitialization
        if (k % 10 == 0 and reinit):
            sign = np.asarray((phi > 0), dtype= np.int) - np.asarray((phi < 0), dtype= np.int)
            [phi_y, phi_x] = np.gradient(phi)
            s = np.sqrt(np.square(phi_x) + np.square(phi_y)) # equation 10
            phi = phi + time_step * sign * (1-s) # equation 10

    return phi

def Chan_Vese_model_update(phi, I, lmda_1, lmda_2, mu, epsilon, timestep):
    # Chan_Vese model is a two-phase level set in this paper
    phase_number = 2

    diracPhi = Dirac(phi, epsilon)

    # update areaTerm
    areaTerm = compute_areaTerm(I, phi, epsilon, phase_number, lmda_1, lmda_2)

    # update lengthTerm
    lengthTerm = compute_lengthTerm(phi, mu)

    phi_update = phi + timestep * diracPhi * (areaTerm + lengthTerm)

    return phi_update

def compute_areaTerm(I, phi, epsilon, phase_number, lmda_1, lmda_2):
    H_phi = Heaviside(phi, epsilon)

    # member function: H(phi) and 1 - H(phi)
    member_function = np.zeros((I.shape[0], I.shape[1], phase_number))
    member_function[:, :, 0] = H_phi
    member_function[:, :, 1] = 1 - H_phi

    # differential of H_phi for each member function
    M0H1 = 1
    M1H1 = -1

    # calculate c1 and c2
    C = update_C(I, member_function, phase_number)

    # compute area term for every phi
    areaTerm = (-1) * (lmda_1 * (I - C[:, :, 0]) * (I - C[:, :, 0]) * M0H1 + lmda_2 * (I - C[:, :, 1]) * (
                I - C[:, :, 1]) * M1H1)

    return areaTerm

def update_C (img, member_function, phase):
    # C : c1 and c2
    C = np.zeros((img.shape[0], img.shape[1], phase))
    for i in range (phase):
        img_sum = np.sum(img * member_function[:, :, i])
        m_sum = np.sum(member_function[:, :, i])
        C[:,:,i] = img_sum / m_sum

    return C

def compute_lengthTerm(phi, mu):
    [phi_y, phi_x] = np.gradient(phi)
    s = np.sqrt(np.square(phi_x) + np.square(phi_y))
    delta = 1e-10
    n_x = phi_x / (s + delta)  # add a small positive number to avoid division by zero
    n_y = phi_y / (s + delta)
    lengthTerm = div(n_x, n_y)

    return mu * lengthTerm
