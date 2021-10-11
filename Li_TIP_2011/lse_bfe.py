# -*- coding: utf-8 -*-

from scipy.ndimage import gaussian_filter
from utils import *

def lse_bfe(u0,Img, b, Ksigma ,KONE, nu, timestep, mu,epsilon, iter_lse):
    u = u0.copy()

    KB1 = gaussian_filter(b, sigma=Ksigma, truncate=2)
    KB2 = gaussian_filter(b*b, sigma=Ksigma, truncate=2)

    C_update = updateC(Img, u, KB1, KB2, epsilon)

    KONE_Img = Img * Img * KONE

    u_update = updateLSF(Img, u, C_update, KONE_Img, KB1, KB2, mu, nu, timestep, epsilon, iter_lse)

    Hu = Heaviside(u, epsilon)

    M = np.zeros((Hu.shape[0], Hu.shape[1], 2))
    M[:,:, 0] = Hu
    M[:,:, 1] =1 - Hu

    b_update = updateB(Img, C_update, M, Ksigma)

    return u_update, b_update, C_update


def updateC(Img, u, Kb1, Kb2, epsilon):
    Hu = Heaviside(u,epsilon)
    M = np.zeros((Hu.shape[0], Hu.shape[1], 2))
    M[:,:, 0]=Hu
    M[:,:, 1]=1 - Hu
    C_new = []
    for i in range (2):
        Nm2 = Kb1 * Img * M[:,:,i]
        Dn2 = Kb2 * M[:,:,i]
        C_new.append(np.sum(Nm2) / np.sum(Dn2))

    return C_new


def updateLSF(Img, u0, C, KONE_Img, KB1, KB2, mu, nu, timestep, epsilon, iter_lse):
    u = u0.copy()
    Hu = Heaviside(u, epsilon)
    M = np.zeros((Hu.shape[0], Hu.shape[1], 2))
    M[:,:, 0]=Hu
    M[:,:, 1]=1 - Hu
    e = np.zeros_like(M)

    for i in range (2):
        e[:,:,i] = KONE_Img - 2 * Img * C[i] * KB1 + C[i] * C[i] * KB2

    for j in range (iter_lse):
        u = neumann_bound_cond(u)
        K = curvature_central(u)
        DiracU = Dirac(u, epsilon)
        ImageTerm = -DiracU * (e[:,:, 0] - e[:,:, 1])
        penalizeTerm=mu*dist_reg_p2(u)
        lengthTerm = nu * DiracU * K
        u = u + timestep * (lengthTerm + penalizeTerm + ImageTerm)

    return u

def updateB(Img, C, M, Ksigma):
    PC1 = np.zeros_like(Img)
    PC2 = np.zeros_like(Img)
    for i in range (2):
        PC1 = PC1 + C[i] * M[:,:, i]
        PC2 = PC2 + C[i] * C[i] * M[:,:, i]

    KNm1 = gaussian_filter(PC1 * Img, sigma= Ksigma, truncate=2)
    KDn1 = gaussian_filter(PC2, sigma= Ksigma, truncate=2)

    b = KNm1 / KDn1

    return  b