# -*- coding: utf-8 -*-

from utils import *

def evolution(phi_0, I, lmda, mu, alfa, epsilon, time_step, iters, potential_function, display = True):
    """
    :param phi_0: initial level set function
    :param I: the input image to be segmented
    :param lmda: weight of the weighted length term
    :param mu: weight of distance regularization term
    :param alfa: weight of the weighted area term
    :param epsilon: width of Dirac Delta function
    :param time_step: time step
    :param iters: number of iterations
    :param potential_function: choice of potential function in distance regularization term.
              As mentioned in the above paper, two choices are provided: potentialFunction='single-well' or
              potentialFunction='double-well', which correspond to the potential functions p1 (single-well)
              and p2 (double-well), respectively.
    :param display: whether display the evolution
    """
    phi = phi_0.copy()
    g = get_g (I)
    for k in range(iters):
        if (display):
            draw_all(phi, I)
        phi = updatePhi(phi, I, epsilon, g, time_step, mu, lmda, alfa, potential_function)

    return phi

def updatePhi (phi_0, I, epsilon, g, time_step, mu, lmda, alfa, potential_function):
    phi = neumann_bound_cond(phi_0)

    # the Chan_Vese module is a two-phase level set
    phase_number = 2
    # update areaTerm
    area_term = compute_areaTerm(I, phi, epsilon, phase_number, lmda_1 = 1, lmda_2 = 1) # balloon/pressure force

    [vy, vx] = np.gradient(g)

    [phi_y, phi_x] = np.gradient(phi)
    s = np.sqrt(np.square(phi_x) + np.square(phi_y))
    delta = 1e-10
    n_x = phi_x / (s + delta)  # add a small positive number to avoid division by zero
    n_y = phi_y / (s + delta)
    curvature = div(n_x, n_y)

    if potential_function == 'single-well':
        dist_reg_term = laplace(phi, mode='nearest') - curvature  # compute distance regularization term in equation (13) with the single-well potential p1.
    elif potential_function == 'double-well':
        dist_reg_term = dist_reg_p2(phi)  # compute the distance regularization term in eqaution (13) with the double-well potential p2.
    else:
        raise Exception('Error: Wrong choice of potential function. Please input the string "single-well" or "double-well" in the drlse_edge function.')
    dirac_phi = dirac(phi, epsilon)

    # area term of DRLSE is replaced by the area term of CV to form the region-based DRLSE
    #area_term = dirac_phi * g  # balloon/pressure force
    edge_term = dirac_phi * (vx * n_x + vy * n_y) + dirac_phi * g * curvature
    phi_update = phi + time_step * (mu * dist_reg_term + lmda * edge_term + alfa * area_term)

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
