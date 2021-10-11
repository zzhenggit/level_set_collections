# -*- coding: utf-8 -*-

from scipy.ndimage import gaussian_filter
from utils import *

def lse_bfe_3Phase(u,Img,b,sigma,KONE, nu,timestep,mu,epsilon,Iter):

    N_class = 3 # % three-phase
    KB1 = gaussian_filter(b, sigma= sigma, truncate=2)
    KB2 = gaussian_filter(b*b, sigma=sigma, truncate=2)

    H1 = Heaviside(u[:,:,0], epsilon)
    H2 = Heaviside(u[:,:,1], epsilon)

    M = np.zeros((Img.shape[0], Img.shape[1], 3))
    M[:, :, 0] = H1 * H2 # % membership function 1
    M[:, :, 1] = H1 * (1 - H2)  # % membership function 2
    M[:, :, 2] = 1 - H1  # % membership function 1

    C_update = updateC(Img, KB1, KB2, M)

    KONE_Img = Img * Img * KONE

    u_update = updateLSF(Img, u, C_update, N_class, KONE_Img, KB1, KB2, mu, nu, timestep, epsilon, Iter)

    b_update = updateB(Img, C_update, M, sigma)

    return u_update, b_update, C_update

def updateLSF(Img,u_0, C, N_class, KONE_Img, KB1, KB2, mu, nu, timestep, epsilon, Iter):
    u = u_0.copy()

    e = np.zeros((Img.shape[0], Img.shape[1], 3))

    for i in range(N_class):
        e[:,:,i] = KONE_Img - 2* Img * C[i] * KB1 + C[i] * C[i] * KB2

    for j in range (Iter):
        Curv = np.zeros((Img.shape[0], Img.shape[1], 2))
        Delta = np.zeros((Img.shape[0], Img.shape[1], 2))
        u[:, :, 0] = neumann_bound_cond(u[:, :, 0])
        Curv[:, :, 0] = curvature_central(u[:, :, 0])
        H1 = Heaviside(u[:, :, 0], epsilon)
        Delta[:, :, 0] = Dirac(u[:, :, 0], epsilon)

        u[:, :, 1] = neumann_bound_cond(u[:, :, 1])
        Curv[:, :, 1] = curvature_central(u[:, :, 1])
        H2 = Heaviside(u[:, :, 1], epsilon)
        Delta[:, :, 1] = Dirac(u[:, :, 1], epsilon)

        A1 = - Delta[:,:, 0] * (e[:,:, 0] * H2 + e[:,:, 1] *(1 - H2) - e[:,:, 2])
        #P1 = mu * dist_reg_p2(u[:,:,0])
        P1 = mu * (laplace(u[:,:,0], mode='nearest') - Curv[:,:,0])

        L1 = nu * Delta[:,:, 0] * Curv[:,:, 0]
        u[:,:, 0] = u[:,:, 0]+timestep * (L1 + P1 + A1) # % update u1

        A2 = - Delta[:,:,1] * H1 *(e[:,:,0]-e[:,:,1])
        #P2 = mu * dist_reg_p2(u[:,:,1])
        P2 = mu * (laplace(u[:, :, 1], mode='nearest') - Curv[:, :, 1])
        L2 = nu * Delta[:,:, 1] * Curv[:,:, 1]
        u[:,:, 1] = u[:,:, 1]+timestep * (L2 + P2 + A2) # % update u2

    return u

def updateC(Img, Kb1, Kb2, M):
    C = []
    for i in range(3):
        N2 = Kb1 * Img * M[:,:, i]
        D2 = Kb2 * M[:,:, i]
        sN2 = np.sum(N2)
        sD2 = np.sum(D2)
        C.append(sN2 / (sD2 + (sD2 == 0)))

    return C

def updateB(Img, C, M,  Ksigma):
    PC1 = np.zeros_like(Img)
    PC2 = np.zeros_like(Img)

    for i in range(3):
        PC1 = PC1 + C[i] * M[:,:, i]
        PC2 = PC2 + C[i] * C[i] * M[:,:, i]

    KNm1 = gaussian_filter(PC1 * Img, sigma= Ksigma, truncate=2)
    KDn1 = gaussian_filter(PC2, sigma= Ksigma, truncate=2)

    b = KNm1 / KDn1

    return  b