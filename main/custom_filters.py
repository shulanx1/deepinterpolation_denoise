# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 16:24:44 2023

@author: xiao208
"""
import numpy as np

def kalman_stack_filter(Img,G=0.8, V=0.05):
    """
    
    Kalman filter with each frame regarded as one measurement
    Parameters
    ----------
    G : scalar
        filter gain
    V : scalar
        estimated variance

    Returns
    -------
    None.

    """
    dimz = len(Img)
    dimx = len(Img[1])
    dimy = len(Img[2])
    Img_s = np.reshape(Img, (dimz, dimx*dimy))

    #initialization
    Ik = Img_s[0] # use the first image as prediction seed
    Ek = V # use the estimated variance

    Img_p = [Img[0]]
    #iteration
    for k in range(dimz-1):
        # correction
        Mk = Img_s[k+1] # current measurement
        K = Ek/(Ek+V)   #kalman gain
        Ik = G*Ik+(1-G)*Mk+K*(Mk-Ik)  # updated correction
        Ek = Ek*(1-K) # updated estimation of variance

        #prediction
        Img_p.append(np.reshape(Ik, (dimx, dimy)))

    return np.asarray(Img_p)


