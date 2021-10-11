# -*- coding: utf-8 -*-

from skimage import measure
from utils import *

def evolution(phi_0, I, mu, epsilon, potential_function, lmda, alfa, time_step, iters, display = True):
    """
    :param phi_0: initial level set function
    :param I: the input image to be segmented
    :param mu: weight of length term
    :param epsilon: width of Dirac Delta function
    :param potential_function: which potential_function to use
    :param lmda: weight of the edge term
    :param alfa: weight of the area term
    :param time_step: time step
    :param iters: number of iterations
    :param display: whether display the evolution
    """
    phi = phi_0.copy()

    g = get_g (I)

    for k in range(iters):
        # display
        if (display):
            plt.ion()
            fig1 = plt.figure(1)
            fig1.clf()
            init_contours1 = measure.find_contours(phi[:,:,0], 0.5)
            init_contours2 = measure.find_contours(phi[:,:,1], 0.5)
            show_image_and_segmentation_all_two(fig1, I, init_contours1,init_contours2)
            fig1.suptitle("demo")
            plt.pause(0.5)
            plt.show()

        phi = multi_phase_Chan_Vese_model_update(phi, I, mu, epsilon, time_step, g, potential_function, lmda, alfa)

    return phi

def multi_phase_Chan_Vese_model_update(phi_0, I, mu, epsilon, time_step, g, potential_function, lmda, alfa):
    # four-phase segmented via two level sets
    phase_number = 4
    phi = phi_0.copy()

    # update area_term of CV
    area_term = compute_areaTerm(phi, I,  epsilon,  phase_number)

    # DRLSE
    [vy, vx] = np.gradient(g)

    for i in range (2):
        [phi_y, phi_x] = np.gradient(phi[:,:,i])
        s = np.sqrt(np.square(phi_x) + np.square(phi_y))
        delta = 1e-10
        n_x = phi_x / (s + delta)  # add a small positive number to avoid division by zero
        n_y = phi_y / (s + delta)
        curvature = div(n_x, n_y)

        if potential_function == 'single-well':
            dist_reg_term = laplace(phi[:,:,i], mode='nearest') - curvature
        elif potential_function == 'double-well':
            dist_reg_term = dist_reg_p2(phi[:,:,i])
        else:
            raise Exception('Error: Wrong choice of potential function. Please input the string "single-well" or "double-well" in the drlse_edge function.')
        dirac_phi = dirac(phi[:,:,i], epsilon)

        edge_term = dirac_phi * (vx * n_x + vy * n_y) + dirac_phi * g * curvature

        phi[:, :, i] = phi[:, :, i] + time_step * (mu * dist_reg_term + lmda * edge_term + alfa * area_term[:,:,i])

    return phi

def compute_areaTerm(phi, img, epsilon, phase_number):
    H1 = Heaviside(phi[:, :, 0], epsilon)
    H2 = Heaviside(phi[:, :, 1], epsilon)

    # member function of each level set
    member_function = np.zeros((img.shape[0], img.shape[1], phase_number))
    member_function[:, :, 0] = H1 * H2
    member_function[:, :, 1] = H1 * (1-H2)
    member_function[:, :, 2] = (1-H1) * H2
    member_function[:, :, 3] = (1-H1) * (1-H2)

    # differential of H1 for each member function
    M0H1 = H2
    M1H1 = 1 - H2
    M2H1 = -H2
    M3H1 = -1 + H2
    # differential of H2 for each member function
    M0H2 = H1
    M1H2 = -H1
    M2H2 = 1-H1
    M3H2 = -1 + H1

    C = update_C(img, member_function, phase_number)

    # compute area term for every phi
    areaTerm = np.zeros_like(phi)
    areaTerm[:,:,0] = (-1) * \
                ((img - C[:, :, 0]) * (img - C[:, :, 0]) * M0H1 + (img - C[:, :, 1]) * (img - C[:, :, 1]) * M1H1 +
                 (img - C[:, :, 2]) * (img - C[:, :, 2]) * M2H1 + (img - C[:, :, 3]) * (img - C[:, :, 3]) * M3H1)

    areaTerm[:,:,1] = (-1) * \
                ((img - C[:, :, 0]) * (img - C[:, :, 0]) * M0H2 + (img - C[:, :, 1]) * (img - C[:, :, 1]) * M1H2 +
                 (img - C[:, :, 2]) * (img - C[:, :, 2]) * M2H2 + (img - C[:, :, 3]) * (img - C[:, :, 3]) * M3H2)

    return areaTerm

def update_C (img, member_function, phase):
    C = np.zeros((img.shape[0], img.shape[1], phase))
    for i in range (phase):
        img_sum = np.sum(img * member_function[:, :, i])
        m_sum = np.sum(member_function[:, :, i])
        C[:,:,i] = img_sum / m_sum
    return C

def compute_lengthTerm_all (phi,mu):
    lengthTerm = np.zeros_like(phi)
    lengthTerm[:, :, 0] = compute_lengthTerm(phi[:, :, 0], mu)
    lengthTerm[:, :, 1] = compute_lengthTerm(phi[:, :, 1], mu)
    return lengthTerm

def compute_lengthTerm(phi, mu):
    [phi_y, phi_x] = np.gradient(phi)
    s = np.sqrt(np.square(phi_x) + np.square(phi_y))
    delta = 1e-10
    n_x = phi_x / (s + delta)  # add a small positive number to avoid division by zero
    n_y = phi_y / (s + delta)
    lengthTerm = div(n_x, n_y)
    return mu * lengthTerm
