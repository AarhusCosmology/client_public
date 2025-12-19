#!/usr/bin/env python3
"""
Compare iteration 9 models from two Gaussian runs against the true Gaussian likelihood.
This script evaluates both surrogate models and the true likelihood on a set of test points,
then computes various error metrics and creates visualizations.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import pickle

textwidth_pts = 440
width_inches = textwidth_pts / 72.27
fontsize = 11 / 1.2 

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': 'cmr10',
    'font.size': fontsize,
    'mathtext.fontset': 'cm',
    'axes.formatter.use_mathtext': True,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.6,
    'ytick.minor.width': 0.6,
    'lines.linewidth': 1.5,
    'patch.linewidth': 0.8,
    'grid.linewidth': 0.5,
})

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from likelihood.base import BaseLikelihood
from likelihood.surrogate import EmulatedLikelihood
from config.config_loader import load_config
import numpy as np


def gaussian_loglkl_planck(
    omega_b, omega_cdm, theta_s_100, ln10_10_A_s, n_s, tau_reio,
    A_cib_217, xi_sz_cib, A_sz, ps_A_100_100, ps_A_143_143, ps_A_143_217,
    ps_A_217_217, ksz_norm, gal545_A_100, gal545_A_143, gal545_A_143_217,
    gal545_A_217, galf_TE_A_100, galf_TE_A_100_143, galf_TE_A_100_217,
    galf_TE_A_143, galf_TE_A_143_217, galf_TE_A_217, calib_100T, calib_217T,
    A_planck
):
    """Gaussian likelihood - same as banana but without the penalty term (only 27 Planck parameters)."""
    mu_planck = np.array([0.02281, 0.118, 1.041, 3.009, 0.9586, 0.05981, 33.29, 0.3325, 5.377, 173.8, 57.1, 29.52, 127.1, 4.046, 5.413, 10.89, 24.27, 91.74, 0.01359, 0.142, 0.5247, 0.2598, 0.8805, 2.113, 1.001, 0.997, 0.9956])
    cov_planck = np.array([
      [1.838452e-08, -5.855070e-08, 6.096961e-09, 3.580711e-07, 1.700680e-07, 2.036630e-07, -7.180646e-05, 8.743827e-07, 2.187592e-05, -4.114408e-04, -2.070200e-04, -5.397157e-05, 1.905007e-05, -4.073485e-05, -1.643431e-06, -6.687701e-06, -1.071515e-05, 2.444881e-06, -4.792953e-08, -2.273091e-08, 1.993867e-07, -1.121586e-07, -4.256584e-07, -1.370766e-06, 2.296628e-09, -9.119030e-10, 4.326507e-09],
      [-5.855070e-08, 9.088668e-07, -2.805494e-08, -3.618013e-06, -2.245863e-06, -2.669441e-06, 1.996329e-04, -9.026381e-07, -6.078604e-05, 1.412000e-03, 7.635860e-04, 3.212141e-04, 1.903030e-05, 9.965589e-05, -1.186369e-05, 2.531657e-05, 4.324023e-05, -3.275044e-05, -5.394656e-07, -2.319351e-07, 7.750278e-07, 8.761921e-07, 1.965984e-06, 7.747312e-06, -8.429232e-09, 2.042284e-09, -2.017848e-07],
      [6.096961e-09, -2.805494e-08, 8.082583e-08, 4.248804e-07, 1.838079e-07, 1.887152e-07, 1.409270e-05, -7.190088e-07, 2.601654e-06, -1.244134e-04, -9.489574e-06, 2.392282e-06, 3.762703e-07, -1.525368e-06, -3.085484e-07, 7.097975e-07, -2.820196e-06, -1.064256e-05, -2.645386e-08, -3.440218e-08, 3.979028e-07, -2.291433e-07, -4.859616e-07, -1.991939e-06, -3.654166e-09, 3.032249e-10, 2.158665e-08],
      [3.580711e-07, -3.618013e-06, 4.248804e-07, 2.145984e-04, 1.372989e-05, 1.019803e-04, -1.989271e-03, 5.058830e-05, 7.604652e-04, -1.532744e-02, -6.314029e-03, 1.113905e-03, 4.487900e-03, -2.539971e-03, -1.070755e-04, -1.137398e-03, -1.622942e-03, 5.012499e-04, 1.721894e-06, -5.089333e-06, 6.511646e-06, -1.497603e-05, -8.417551e-06, -6.502149e-05, 7.691228e-10, -3.128372e-08, 9.098536e-06],
      [1.700680e-07, -2.245863e-06, 1.838079e-07, 1.372989e-05, 1.465672e-05, 9.593658e-06, -4.513114e-03, 9.370318e-05, 9.506377e-04, -2.395196e-02, -6.453758e-03, 1.764920e-03, 5.324884e-03, -2.325031e-03, 3.337894e-04, -3.810500e-04, -3.558027e-04, 1.366334e-03, 5.979286e-07, 8.538959e-07, -4.893338e-06, 8.193760e-07, -3.939617e-06, -1.185534e-05, -5.117653e-08, -5.509131e-08, 5.876584e-07],
      [2.036630e-07, -2.669441e-06, 1.887152e-07, 1.019803e-04, 9.593658e-06, 5.560892e-05, -1.159082e-03, -2.393317e-06, 4.329611e-04, -1.013001e-02, -4.970747e-03, -1.471531e-03, -5.848395e-04, -9.384664e-04, 1.467935e-04, -3.353150e-04, -6.477103e-04, -4.564894e-04, 4.244739e-07, -2.611594e-06, 3.348812e-06, -6.027112e-06, -4.397122e-06, -4.313478e-05, -5.442858e-08, -2.990148e-08, -9.733190e-07],
      [-7.180646e-05, 1.996329e-04, 1.409270e-05, -1.989271e-03, -4.513114e-03, -1.159082e-03, 4.369913e+01, -4.093948e-01, -7.472189e-01, 3.607074e+01, 1.326719e+00, -1.343715e+01, -5.399643e+01, -2.195230e-01, 9.072955e-02, 4.866182e-01, -5.690536e+00, -3.253416e+01, 2.280814e-03, -4.237131e-04, 7.827614e-04, -2.125696e-03, -7.520311e-04, -3.377007e-03, -1.507015e-04, 7.096093e-04, 6.891280e-04],
      [8.743827e-07, -9.026381e-07, -7.190088e-07, 5.058830e-05, 9.370318e-05, -2.393317e-06, -4.093948e-01, 8.153652e-02, -3.945583e-02, -6.064150e-01, 9.296235e-01, 1.714510e+00, 6.733326e-01, 4.011858e-02, -3.997833e-03, 1.076844e-02, 1.130743e-01, 1.545585e-01, -1.055693e-04, 6.945283e-05, -2.186106e-05, 2.427217e-04, -3.902195e-05, -5.605277e-05, 2.341670e-06, -5.171045e-06, 6.942143e-06],
      [2.187592e-05, -6.078604e-05, 2.601654e-06, 7.604652e-04, 9.506377e-04, 4.329611e-04, -7.472189e-01, -3.945583e-02, 3.872661e+00, -3.091765e+01, -3.278724e+00, 7.728797e+00, 5.318349e+00, -2.766041e+00, -2.317309e-01, 1.246514e-01, 2.616894e+00, 3.216839e+00, -5.885545e-05, -5.983675e-04, -1.876935e-04, -4.177155e-04, -3.967020e-03, -5.783100e-03, 9.504072e-05, -3.052376e-05, 3.247259e-06],
      [-4.114408e-04, 1.412000e-03, -1.244134e-04, -1.532744e-02, -2.395196e-02, -1.013001e-02, 3.607074e+01, -6.064150e-01, -3.091765e+01, 7.817827e+02, 6.868638e+01, -5.743992e+01, -6.301325e+01, 1.489491e+01, -6.565124e+00, 5.723957e+00, -1.236890e+01, -2.462541e+01, -1.817985e-03, 3.069930e-03, 2.820445e-02, 3.007281e-03, 3.181726e-02, 2.242627e-02, 3.478651e-03, 6.387539e-04, 3.277322e-03],
      [-2.070200e-04, 7.635860e-04, -9.489574e-06, -6.314029e-03, -6.453758e-03, -4.970747e-03, 1.326719e+00, 9.296235e-01, -3.278724e+00, 6.868638e+01, 6.117934e+01, 4.022831e+01, 1.549531e+01, -4.866975e+00, -6.124554e-01, -1.342400e+00, 3.377971e+00, 4.482002e+00, 1.333083e-03, 2.857762e-03, -3.741254e-03, 5.231590e-04, 3.111790e-03, 4.769279e-02, 1.529862e-05, 9.525768e-05, 9.432686e-04],
      [-5.397157e-05, 3.212141e-04, 2.392282e-06, 1.113905e-03, 1.764920e-03, -1.471531e-03, -1.343715e+01, 1.714510e+00, 7.728797e+00, -5.743992e+01, 4.022831e+01, 8.566116e+01, 5.376889e+01, -1.197730e+01, -9.389431e-01, -7.266080e-01, 5.163288e+00, 8.764122e+00, -2.376079e-03, -1.789589e-03, -1.594252e-03, 7.853313e-04, -1.203247e-02, 5.495060e-04, 3.272048e-04, -3.799806e-04, 8.231180e-04],
      [1.905007e-05, 1.903030e-05, 3.762703e-07, 4.487900e-03, 5.324884e-03, -5.848395e-04, -5.399643e+01, 6.733326e-01, 5.318349e+00, -6.301325e+01, 1.549531e+01, 5.376889e+01, 1.025537e+02, -9.293540e+00, -6.537521e-01, -1.229966e+00, 5.444676e+00, 3.266589e+01, -3.525033e-03, -4.916286e-04, -2.603240e-03, 3.455467e-03, -2.902487e-03, 1.426857e-02, 3.719083e-04, -1.317499e-03, 1.327616e-03],
      [-4.073485e-05, 9.965589e-05, -1.525368e-06, -2.539971e-03, -2.325031e-03, -9.384664e-04, -2.195230e-01, 4.011858e-02, -2.766041e+00, 1.489491e+01, -4.866975e+00, -1.197730e+01, -9.293540e+00, 5.939199e+00, 1.454696e-01, -1.525418e-01, -2.080716e+00, -2.289480e+00, 4.475404e-05, 2.752792e-04, 1.730246e-03, -4.251593e-04, 3.380746e-03, -2.202818e-03, -6.861400e-05, -1.382594e-05, -5.252022e-05],
      [-1.643431e-06, -1.186369e-05, -3.085484e-07, -1.070755e-04, 3.337894e-04, 1.467935e-04, 9.072955e-02, -3.997833e-03, -2.317309e-01, -6.565124e+00, -6.124554e-01, -9.389431e-01, -6.537521e-01, 1.454696e-01, 3.332873e+00, 5.595878e-01, 1.344987e-01, 7.580292e-02, -9.142100e-05, 1.459505e-04, -3.434092e-03, 3.880520e-04, -4.324599e-05, -2.637324e-03, 1.775477e-04, -1.098972e-05, -4.634541e-05],
      [-6.687701e-06, 2.531657e-05, 7.097975e-07, -1.137398e-03, -3.810500e-04, -3.353150e-04, 4.866182e-01, 1.076844e-02, 1.246514e-01, 5.723957e+00, -1.342400e+00, -7.266080e-01, -1.229966e+00, -1.525418e-01, 5.595878e-01, 3.194709e+00, 2.669024e+00, 2.624229e+00, -2.109322e-04, -4.719761e-04, -3.132446e-04, 9.150606e-04, -1.053721e-03, -8.675063e-03, -1.742823e-04, -5.221071e-05, 1.923082e-05],
      [-1.071515e-05, 4.324023e-05, -2.820196e-06, -1.622942e-03, -3.558027e-04, -6.477103e-04, -5.690536e+00, 1.130743e-01, 2.616894e+00, -1.236890e+01, 3.377971e+00, 5.163288e+00, 5.444676e+00, -2.080716e+00, 1.344987e-01, 2.669024e+00, 1.049552e+01, 1.874051e+01, 8.840301e-04, 4.126084e-04, -6.420726e-04, 1.133924e-03, -2.215955e-03, 1.807058e-03, -3.542913e-05, 3.926255e-04, 1.177300e-05],
      [2.444881e-06, -3.275044e-05, -1.064256e-05, 5.012499e-04, 1.366334e-03, -4.564894e-04, -3.253416e+01, 1.545585e-01, 3.216839e+00, -2.462541e+01, 4.482002e+00, 8.764122e+00, 3.266589e+01, -2.289480e+00, 7.580292e-02, 2.624229e+00, 1.874051e+01, 5.237645e+01, 1.300609e-03, 5.475081e-04, 1.443485e-03, 7.750407e-04, -2.655393e-03, 1.146156e-02, 2.006470e-05, 9.350594e-04, 7.803558e-04],
      [-4.792953e-08, -5.394656e-07, -2.645386e-08, 1.721894e-06, 5.979286e-07, 4.244739e-07, 2.280814e-03, -1.055693e-04, -5.885545e-05, -1.817985e-03, 1.333083e-03, -2.376079e-03, -3.525033e-03, 4.475404e-05, -9.142100e-05, -2.109322e-04, 8.840301e-04, 1.300609e-03, 1.421964e-03, 2.549392e-04, 5.835476e-05, -3.719333e-04, -9.409117e-05, -1.536822e-04, -5.846517e-08, 2.506406e-07, 1.004549e-06],
      [-2.273091e-08, -2.319351e-07, -3.440218e-08, -5.089333e-06, 8.538959e-07, -2.611594e-06, -4.237131e-04, 6.945283e-05, -5.983675e-04, 3.069930e-03, 2.857762e-03, -1.789589e-03, -4.916286e-04, 2.752792e-04, 1.459505e-04, -4.719761e-04, 4.126084e-04, 5.475081e-04, 2.549392e-04, 8.656543e-04, 1.892216e-05, 7.554639e-04, 5.102508e-05, -7.696566e-04, 9.910568e-08, -3.713433e-08, 2.081442e-07],
      [1.993867e-07, 7.750278e-07, 3.979028e-07, 6.511646e-06, -4.893338e-06, 3.348812e-06, 7.827614e-04, -2.186106e-05, -1.876935e-04, 2.820445e-02, -3.741254e-03, -1.594252e-03, -2.603240e-03, 1.730246e-03, -3.434092e-03, -3.132446e-04, -6.420726e-04, 1.443485e-03, 5.835476e-05, 1.892216e-05, 7.235660e-03, -3.592855e-04, 6.531163e-04, 4.435894e-03, 4.559401e-07, 3.368646e-07, -1.912867e-06],
      [-1.121586e-07, 8.761921e-07, -2.291433e-07, -1.497603e-05, 8.193760e-07, -6.027112e-06, -2.125696e-03, 2.427217e-04, -4.177155e-04, 3.007281e-03, 5.231590e-04, 7.853313e-04, 3.455467e-03, -4.251593e-04, 3.880520e-04, 9.150606e-04, 1.133924e-03, 7.750407e-04, -3.719333e-04, 7.554639e-04, -3.592855e-04, 2.955564e-03, 5.220351e-04, -1.187350e-03, -9.310233e-09, 9.701572e-08, -2.196519e-06],
      [-4.256584e-07, 1.965984e-06, -4.859616e-07, -8.417551e-06, -3.939617e-06, -4.397122e-06, -7.520311e-04, -3.902195e-05, -3.967020e-03, 3.181726e-02, 3.111790e-03, -1.203247e-02, -2.902487e-03, 3.380746e-03, -4.324599e-05, -1.053721e-03, -2.215955e-03, -2.655393e-03, -9.409117e-05, 5.102508e-05, 6.531163e-04, 5.220351e-04, 6.446572e-03, 7.691337e-03, 8.558279e-07, 8.007655e-08, -4.759224e-07],
      [-1.370766e-06, 7.747312e-06, -1.991939e-06, -6.502149e-05, -1.185534e-05, -4.313478e-05, -3.377007e-03, -5.605277e-05, -5.783100e-03, 2.242627e-02, 4.769279e-02, 5.495060e-04, 1.426857e-02, -2.202818e-03, -2.637324e-03, -8.675063e-03, 1.807058e-03, 1.146156e-02, -1.536822e-04, -7.696566e-04, 4.435894e-03, -1.187350e-03, 7.691337e-03, 7.149201e-02, 8.404333e-07, -8.667763e-08, 6.207874e-06],
      [2.296628e-09, -8.429232e-09, -3.654166e-09, 7.691228e-10, -5.117653e-08, -5.442858e-08, -1.507015e-04, 2.341670e-06, 9.504072e-05, 3.478651e-03, 1.529862e-05, 3.272048e-04, 3.719083e-04, -6.861400e-05, 1.775477e-04, -1.742823e-04, -3.542913e-05, 2.006470e-05, -5.846517e-08, 9.910568e-08, 4.559401e-07, -9.310233e-09, 8.558279e-07, 8.404333e-07, 3.621674e-07, 2.961786e-09, -5.096304e-09],
      [-9.119030e-10, 2.042284e-09, 3.032249e-10, -3.128372e-08, -5.509131e-08, -2.990148e-08, 7.096093e-04, -5.171045e-06, -3.052376e-05, 6.387539e-04, 9.525768e-05, -3.799806e-04, -1.317499e-03, -1.382594e-05, -1.098972e-05, -5.221071e-05, 3.926255e-04, 9.350594e-04, 2.506406e-07, -3.713433e-08, 3.368646e-07, 9.701572e-08, 8.007655e-08, -8.667763e-08, 2.961786e-09, 3.846599e-07, 1.004939e-08],
      [4.326507e-09, -2.017848e-07, 2.158665e-08, 9.098536e-06, 5.876584e-07, -9.733190e-07, 6.891280e-04, 6.942143e-06, 3.247259e-06, 3.277322e-03, 9.432686e-04, 8.231180e-04, 1.327616e-03, -5.252022e-05, -4.634541e-05, 1.923082e-05, 1.177300e-05, 7.803558e-04, 1.004549e-06, 2.081442e-07, -1.912867e-06, -2.196519e-06, -4.759224e-07, 6.207874e-06, -5.096304e-09, 1.004939e-08, 5.788216e-06]
    ])
    
    inv_cov = np.linalg.inv(cov_planck)
    sign, logdet = np.linalg.slogdet(cov_planck)
    
    x_main = np.array([
        omega_b, omega_cdm, theta_s_100, ln10_10_A_s, n_s, tau_reio,
        A_cib_217, xi_sz_cib, A_sz, ps_A_100_100, ps_A_143_143, ps_A_143_217,
        ps_A_217_217, ksz_norm, gal545_A_100, gal545_A_143, gal545_A_143_217,
        gal545_A_217, galf_TE_A_100, galf_TE_A_100_143, galf_TE_A_100_217,
        galf_TE_A_143, galf_TE_A_143_217, galf_TE_A_217, calib_100T, calib_217T,
        A_planck
    ])
    
    diff = x_main - mu_planck
    quad = np.dot(diff, np.dot(inv_cov, diff))
    const = -0.5 * (27 * np.log(2 * np.pi) + logdet)
    loglkl_main = const - 0.5 * quad
    
    return loglkl_main


class GaussianLikelihood(BaseLikelihood):
    """Direct wrapper for Gaussian likelihood - no Cobaya overhead."""
    
    def __init__(self):
        super().__init__()
        self._setup_parameters()
    
    def _setup_parameters(self):
        """Setup parameter definitions from gaussian_planck.yaml."""
        param_defs = {
            'omega_b': {'range': [0.005, 0.1], 'initial': 0.02281, 'sigma': 0.0001773, 'label': r'\Omega_b h^2'},
            'omega_cdm': {'range': [0.001, 0.99], 'initial': 0.118, 'sigma': 0.0008154, 'label': r'\Omega_{cdm} h^2'},
            'theta_s_100': {'range': [0.5, 10], 'initial': 1.041, 'sigma': 0.0002518, 'label': r'100\,\theta_\mathrm{s}'},
            'ln10_10_A_s': {'range': [1.61, 3.91], 'initial': 3.009, 'sigma': 0.01124, 'label': r'\ln(10^{10}A_s)'},
            'n_s': {'range': [0.8, 1.2], 'initial': 0.9586, 'sigma': 0.004295, 'label': r'n_s'},
            'tau_reio': {'range': [0.01, 0.8], 'initial': 0.05981, 'sigma': 0.003923, 'label': r'\tau_\mathrm{reio}'},
            'A_cib_217': {'range': [0.0, 200.0], 'initial': 33.29, 'sigma': 8.93, 'label': r'A^\mathrm{CIB}_{217}'},
            'xi_sz_cib': {'range': [0.0, 1.0], 'initial': 0.3325, 'sigma': 0.242, 'label': r'\xi_{\mathrm{tSZ}\times\mathrm{CIB}}'},
            'A_sz': {'range': [0.0, 10.0], 'initial': 5.377, 'sigma': 2.097, 'label': r'A_\mathrm{tSZ}'},
            'ps_A_100_100': {'range': [0.0, 400.0], 'initial': 173.8, 'sigma': 26.43, 'label': r'A^{\mathrm{PS}}_{100\times100}'},
            'ps_A_143_143': {'range': [0.0, 400.0], 'initial': 57.1, 'sigma': 10.07, 'label': r'A^{\mathrm{PS}}_{143\times143}'},
            'ps_A_143_217': {'range': [0.0, 400.0], 'initial': 29.52, 'sigma': 5.415, 'label': r'A^{\mathrm{PS}}_{143\times217}'},
            'ps_A_217_217': {'range': [0.0, 400.0], 'initial': 127.1, 'sigma': 9.465, 'label': r'A^{\mathrm{PS}}_{217\times217}'},
            'ksz_norm': {'range': [0.0, 10.0], 'initial': 4.046, 'sigma': 2.234, 'label': r'A_\mathrm{kSZ}'},
            'gal545_A_100': {'range': [0.0, 50.0], 'initial': 5.413, 'sigma': 2.238, 'label': r'A^{\mathrm{gal},545}_{100}'},
            'gal545_A_143': {'range': [0.0, 50.0], 'initial': 10.89, 'sigma': 1.394, 'label': r'A^{\mathrm{gal},545}_{143}'},
            'gal545_A_143_217': {'range': [0.0, 100.0], 'initial': 24.27, 'sigma': 3.168, 'label': r'A^{\mathrm{gal},545}_{143\times217}'},
            'gal545_A_217': {'range': [0.0, 400.0], 'initial': 91.74, 'sigma': 6.017, 'label': r'A^{\mathrm{gal},545}_{217}'},
            'galf_TE_A_100': {'range': [0.0, 10.0], 'initial': 0.01359, 'sigma': 0.03845, 'label': r'A^{\mathrm{gal},TE}_{100}'},
            'galf_TE_A_100_143': {'range': [0.0, 10.0], 'initial': 0.142, 'sigma': 0.03303, 'label': r'A^{\mathrm{gal},TE}_{100\times143}'},
            'galf_TE_A_100_217': {'range': [0.0, 10.0], 'initial': 0.5247, 'sigma': 0.06606, 'label': r'A^{\mathrm{gal},TE}_{100\times217}'},
            'galf_TE_A_143': {'range': [0.0, 10.0], 'initial': 0.2598, 'sigma': 0.06623, 'label': r'A^{\mathrm{gal},TE}_{143}'},
            'galf_TE_A_143_217': {'range': [0.0, 10.0], 'initial': 0.8805, 'sigma': 0.09459, 'label': r'A^{\mathrm{gal},TE}_{143\times217}'},
            'galf_TE_A_217': {'range': [0.0, 10.0], 'initial': 2.113, 'sigma': 0.2954, 'label': r'A^{\mathrm{gal},TE}_{217}'},
            'calib_100T': {'range': [0.0, 3.0], 'initial': 1.001, 'sigma': 0.0007192, 'label': r'c^{T}_{100}'},
            'calib_217T': {'range': [0.0, 3.0], 'initial': 0.997, 'sigma': 0.0005374, 'label': r'c^{T}_{217}'},
            'A_planck': {'range': [0.9, 1.1], 'initial': 0.9956, 'sigma': 0.002401, 'label': r'A_\mathrm{Planck}'},
        }
        
        self.param['varying'] = param_defs
    
    def _loglkl(self, position: dict) -> float:
        """Call Gaussian likelihood function directly."""
        return gaussian_loglkl_planck(**position)
    
    def logprior(self, position: dict) -> float:
        """Uniform prior within bounds."""
        return self.log_uniform_prior(position)
    
    def get_parameter_info(self) -> dict:
        """Return parameter info."""
        return self.param


def load_model_and_likelihood(run_dir, iteration=9):
    """Load surrogate model and true likelihood for a given run."""
    run_path = Path(run_dir)
    
    yaml_files = list(run_path.glob('*.yaml'))
    if not yaml_files:
        raise FileNotFoundError(f"No YAML config found in {run_dir}")
    config_yaml = yaml_files[0]

    cfg = load_config(str(config_yaml))
    
    true_likelihood = GaussianLikelihood()
    if hasattr(cfg, 'n_std') and cfg.n_std is not None:
        true_likelihood.restrict_prior(n_std=cfg.n_std)

    model_path = run_path / f"trained_models/trained_model_it_{iteration}.keras"
    x_scaler_path = run_path / f"scalers/x_scaler_it_{iteration}.pkl"
    y_scaler_path = run_path / f"scalers/y_scaler_it_{iteration}.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    surrogate = EmulatedLikelihood(
        str(model_path),
        str(x_scaler_path),
        str(y_scaler_path),
        true_likelihood=true_likelihood
    )
    
    return surrogate, true_likelihood, cfg


def load_chain_samples(chain_dir='gaussian_planck', max_samples=None, seed=42):
    """Load samples from chain files in the specified directory.
    
    Args:
        chain_dir: Directory containing chain files (*.txt)
        max_samples: Maximum number of samples to load (None = all)
        seed: Random seed for subsampling
    
    Returns:
        samples: Array of shape (n_samples, n_params) with parameter values
    """
    chain_path = Path(chain_dir)
    chain_files = sorted(chain_path.glob('*.txt'))
    
    if not chain_files:
        raise FileNotFoundError(f"No chain files found in {chain_dir}")
    
    print(f"Loading chain samples from {len(chain_files)} files in {chain_dir}...")
    
    all_samples = []
    for chain_file in chain_files:
        if not chain_file.stem.isdigit():
            continue
        
        data = np.loadtxt(chain_file, comments='#')
        params = data[:, 2:29]
        all_samples.append(params)
    
    all_samples = np.vstack(all_samples)
    print(f"  Loaded {len(all_samples)} samples total")

    if max_samples is not None and len(all_samples) > max_samples:
        np.random.seed(seed)
        indices = np.random.choice(len(all_samples), max_samples, replace=False)
        all_samples = all_samples[indices]
        print(f"  Subsampled to {len(all_samples)} samples")
    
    return all_samples


def generate_test_points(likelihood, n_samples=1000, strategy='lhs', seed=42):
    """Generate test points from the prior."""
    np.random.seed(seed)
    
    param_names = likelihood.varying_param_names
    n_params = len(param_names)
    prior_bounds = likelihood.get_prior_bounds()
    
    if strategy == 'lhs':
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=n_params, seed=seed)
        unit_samples = sampler.random(n=n_samples)
        
        samples = np.zeros_like(unit_samples)
        for i, param_name in enumerate(param_names):
            lower, upper = prior_bounds[param_name]
            samples[:, i] = lower + unit_samples[:, i] * (upper - lower)
    
    elif strategy == 'grid':
        n_per_dim = int(np.ceil(n_samples ** (1.0 / n_params)))
        samples = np.zeros((n_per_dim ** n_params, n_params))
        
        grids = []
        for param_name in param_names:
            lower, upper = prior_bounds[param_name]
            grids.append(np.linspace(lower, upper, n_per_dim))
        
        mesh = np.meshgrid(*grids)
        for i in range(n_params):
            samples[:, i] = mesh[i].ravel()
        
        if len(samples) > n_samples:
            indices = np.random.choice(len(samples), n_samples, replace=False)
            samples = samples[indices]
    
    else:
        samples = np.zeros((n_samples, n_params))
        for i, param_name in enumerate(param_names):
            lower, upper = prior_bounds[param_name]
            samples[:, i] = np.random.uniform(lower, upper, n_samples)
    
    return samples


def evaluate_models(samples, true_likelihood, surrogate1, surrogate2, use_parallel=False):
    """Evaluate true likelihood and both surrogates on test points."""
    n_samples = len(samples)
    
    print(f"Evaluating {n_samples} test points...")
    
    print("  Evaluating true likelihood...")
    if use_parallel:
        from utils.mpi_utils import is_mpi_available, parallel_evaluate_likelihood
        param_names = true_likelihood.varying_param_names
        likelihood_func = lambda x: true_likelihood.loglkl({name: float(x[j]) for j, name in enumerate(param_names)})
        true_loglkls = parallel_evaluate_likelihood(samples, likelihood_func, description="true likelihood")
    else:
        param_names = true_likelihood.varying_param_names
        true_loglkls = np.array([
            true_likelihood.loglkl({name: float(samples[i, j]) for j, name in enumerate(param_names)})
            for i in range(n_samples)
        ])
    
    print("  Evaluating surrogate 1...")
    surr1_loglkls = surrogate1.predict(samples)
    
    print("  Evaluating surrogate 2...")
    surr2_loglkls = surrogate2.predict(samples)
    
    return true_loglkls, surr1_loglkls, surr2_loglkls


def compute_errors(true_vals, pred_vals):
    """Compute various error metrics."""
    abs_errors = np.abs(pred_vals - true_vals)
    squared_errors = (pred_vals - true_vals) ** 2
    
    finite_mask = np.isfinite(true_vals) & np.isfinite(pred_vals)
    
    if not np.any(finite_mask):
        return {
            'mae': np.nan,
            'rmse': np.nan,
            'max_abs_error': np.nan,
            'median_abs_error': np.nan,
            'relative_errors': np.array([]),
            'abs_errors': np.array([])
        }
    
    true_finite = true_vals[finite_mask]
    pred_finite = pred_vals[finite_mask]
    abs_errors_finite = abs_errors[finite_mask]
    squared_errors_finite = squared_errors[finite_mask]
    
    relative_errors = np.where(
        np.abs(true_finite) > 1e-10,
        abs_errors_finite / np.abs(true_finite),
        np.nan
    )
    
    metrics = {
        'mae': np.mean(abs_errors_finite),
        'rmse': np.sqrt(np.mean(squared_errors_finite)),
        'max_abs_error': np.max(abs_errors_finite),
        'median_abs_error': np.median(abs_errors_finite),
        'mean_relative_error': np.nanmean(relative_errors),
        'median_relative_error': np.nanmedian(relative_errors),
        'abs_errors': abs_errors_finite,
        'relative_errors': relative_errors[np.isfinite(relative_errors)]
    }
    
    return metrics


def print_comparison_table(errors1, errors2, run1_name, run2_name):
    """Print a comparison table of error metrics."""
    print("\n" + "="*80)
    print("ERROR METRICS COMPARISON")
    print("="*80)
    print(f"\n{'Metric':<30} {run1_name:<20} {run2_name:<20}")
    print("-"*80)
    
    metrics = [
        ('Mean Absolute Error', 'mae'),
        ('Root Mean Square Error', 'rmse'),
        ('Max Absolute Error', 'max_abs_error'),
        ('Median Absolute Error', 'median_abs_error'),
        ('Mean Relative Error (%)', 'mean_relative_error'),
        ('Median Relative Error (%)', 'median_relative_error'),
    ]
    
    for label, key in metrics:
        val1 = errors1[key]
        val2 = errors2[key]
        
        if 'relative' in key.lower():
            val1_str = f"{val1*100:.4f}%" if not np.isnan(val1) else "N/A"
            val2_str = f"{val2*100:.4f}%" if not np.isnan(val2) else "N/A"
        else:
            val1_str = f"{val1:.6f}" if not np.isnan(val1) else "N/A"
            val2_str = f"{val2:.6f}" if not np.isnan(val2) else "N/A"
        
        print(f"{label:<30} {val1_str:<20} {val2_str:<20}")
        
        if not np.isnan(val1) and not np.isnan(val2):
            if val1 < val2:
                print(f"{'':>30} {'✓ BETTER':<20} {'':<20}")
            elif val2 < val1:
                print(f"{'':>30} {'':<20} {'✓ BETTER':<20}")
    
    print("="*80)


def create_comparison_plots(true_loglkls, surr1_loglkls, surr2_loglkls, 
                           errors1, errors2, run1_name, run2_name, output_dir,
                           kappa_sigma1=None, kappa_sigma2=None,
                           true_loglkls_chain=None, surr1_loglkls_chain=None, surr2_loglkls_chain=None,
                           max_points_per_bin=500, n_bins=100):
    """Create residual plot comparing both models in chi-squared space.
    
    Args:
        kappa_sigma1, kappa_sigma2: kappa_sigma values from configs for legend labels
        true_loglkls_chain, surr1_loglkls_chain, surr2_loglkls_chain: Optional chain sample evaluations for left panel
        max_points_per_bin: Maximum number of points to plot per chi^2 bin (density cap)
        n_bins: Number of bins to use for density-based rejection
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    true_chi2 = -2 * true_loglkls
    surr1_chi2 = -2 * surr1_loglkls
    surr2_chi2 = -2 * surr2_loglkls
    
    finite_mask = np.isfinite(true_chi2) & np.isfinite(surr1_chi2) & np.isfinite(surr2_chi2)
    true_finite = true_chi2[finite_mask]
    surr1_finite = surr1_chi2[finite_mask]
    surr2_finite = surr2_chi2[finite_mask]
    
    print(f"  Applying density-based downsampling (max {max_points_per_bin} points per bin)...")
    chi2_min, chi2_max = np.min(true_finite), np.max(true_finite)
    bin_edges = np.linspace(chi2_min, chi2_max, n_bins + 1)
    bin_indices = np.digitize(true_finite, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    keep_mask = np.zeros(len(true_finite), dtype=bool)
    last_full_bin = -1
    for bin_idx in range(n_bins):
        bin_mask = (bin_indices == bin_idx)
        n_in_bin = np.sum(bin_mask)
        
        if n_in_bin > max_points_per_bin:
            bin_point_indices = np.where(bin_mask)[0]
            selected_indices = np.random.choice(bin_point_indices, max_points_per_bin, replace=False)
            keep_mask[selected_indices] = True
            last_full_bin = bin_idx
        else:
            keep_mask[bin_mask] = True
    
    if last_full_bin >= 0:
        chi2_upper = bin_edges[last_full_bin + 1]
    else:
        chi2_upper = chi2_max
    
    true_plot = true_finite[keep_mask]
    surr1_plot = surr1_finite[keep_mask]
    surr2_plot = surr2_finite[keep_mask]
    
    print(f"  Downsampled from {len(true_finite)} to {len(true_plot)} points for plotting")
    print(f"  X-axis upper limit set to χ² = {chi2_upper:.2f} (last full bin)")
    
    has_chain_data = (true_loglkls_chain is not None and 
                      surr1_loglkls_chain is not None and 
                      surr2_loglkls_chain is not None)
    
    if has_chain_data:
        fig, (ax_chain, ax_random) = plt.subplots(1, 2, figsize=(width_inches * 2, width_inches * 0.45))
        
        true_chi2_chain = -2 * true_loglkls_chain
        surr1_chi2_chain = -2 * surr1_loglkls_chain
        surr2_chi2_chain = -2 * surr2_loglkls_chain
        
        finite_mask_chain = np.isfinite(true_chi2_chain) & np.isfinite(surr1_chi2_chain) & np.isfinite(surr2_chi2_chain)
        true_finite_chain = true_chi2_chain[finite_mask_chain]
        surr1_finite_chain = surr1_chi2_chain[finite_mask_chain]
        surr2_finite_chain = surr2_chi2_chain[finite_mask_chain]
        
        chi2_upper_chain = np.percentile(true_finite_chain, 99)
        
        print(f"  Applying density-based downsampling to chain data (max {max_points_per_bin} points per bin)...")
        chi2_min_chain = np.min(true_finite_chain)
        bin_edges_chain = np.linspace(chi2_min_chain, chi2_upper_chain, n_bins + 1)
        bin_indices_chain = np.digitize(true_finite_chain, bin_edges_chain) - 1
        bin_indices_chain = np.clip(bin_indices_chain, 0, n_bins - 1)
        
        keep_mask_chain = np.zeros(len(true_finite_chain), dtype=bool)
        for bin_idx in range(n_bins):
            bin_mask = (bin_indices_chain == bin_idx)
            n_in_bin = np.sum(bin_mask)
            
            if n_in_bin > max_points_per_bin:
                bin_point_indices = np.where(bin_mask)[0]
                selected_indices = np.random.choice(bin_point_indices, max_points_per_bin, replace=False)
                keep_mask_chain[selected_indices] = True
            else:
                keep_mask_chain[bin_mask] = True
        
        true_plot_chain = true_finite_chain[keep_mask_chain]
        surr1_plot_chain = surr1_finite_chain[keep_mask_chain]
        surr2_plot_chain = surr2_finite_chain[keep_mask_chain]
        
        print(f"  Downsampled chain data from {len(true_finite_chain)} to {len(true_plot_chain)} points")
        
        residuals1_chain = surr1_plot_chain - true_plot_chain
        residuals2_chain = surr2_plot_chain - true_plot_chain
        
        in_range_chain = true_plot_chain <= chi2_upper_chain
        true_inrange_chain = true_plot_chain[in_range_chain]
        residuals1_inrange_chain = residuals1_chain[in_range_chain]
        residuals2_inrange_chain = residuals2_chain[in_range_chain]
        
        label1 = rf'$\kappa_\sigma = {kappa_sigma1}$' if kappa_sigma1 is not None else run1_name
        label2 = rf'$\kappa_\sigma = {kappa_sigma2}$' if kappa_sigma2 is not None else run2_name
        
        ax_chain.scatter(true_inrange_chain, residuals2_inrange_chain, alpha=0.4, s=12, label=label2, color='C1', rasterized=True)
        ax_chain.scatter(true_inrange_chain, residuals1_inrange_chain, alpha=0.4, s=12, label=label1, color='C0', rasterized=True)
        ax_chain.axhline(y=0, color='black', linestyle='--', lw=1, alpha=0.5, label='Perfect prediction')
        ax_chain.set_xlim(right=chi2_upper_chain)
        ax_chain.set_xlabel(r'$\chi^2_{\mathrm{true}}$')
        ax_chain.set_ylabel(r'$\chi^2_{\mathrm{surr}} - \chi^2_{\mathrm{true}}$')
        ax_chain.legend(loc='best')
        ax_chain.grid(alpha=0.3, linewidth=0.5)
        
        ax = ax_random
    else:
        fig, ax = plt.subplots(1, 1, figsize=(width_inches, width_inches * 0.5))
    
    residuals1 = surr1_plot - true_plot
    residuals2 = surr2_plot - true_plot
    
    in_range = true_plot <= chi2_upper
    true_inrange = true_plot[in_range]
    residuals1_inrange = residuals1[in_range]
    residuals2_inrange = residuals2[in_range]
    
    label1 = rf'$\kappa_\sigma = {kappa_sigma1}$' if kappa_sigma1 is not None else run1_name
    label2 = rf'$\kappa_\sigma = {kappa_sigma2}$' if kappa_sigma2 is not None else run2_name
    
    ax.scatter(true_inrange, residuals1_inrange, alpha=0.4, s=12, label=label1, color='C0', rasterized=True)
    ax.scatter(true_inrange, residuals2_inrange, alpha=0.4, s=12, label=label2, color='C1', rasterized=True)
    ax.axhline(y=0, color='black', linestyle='--', lw=1, alpha=0.5, label='Perfect prediction')
    
    ax.set_xlim(right=chi2_upper)
    ax.set_xlabel(r'$\chi^2_{\mathrm{true}}$')
    if not has_chain_data:
        ax.set_ylabel(r'$\chi^2_{\mathrm{surr}} - \chi^2_{\mathrm{true}}$')
    ax.legend(loc='best')
    ax.grid(alpha=0.3, linewidth=0.5)
    
    if has_chain_data:
        fig.subplots_adjust(left=0.035, right=0.98, bottom=0.16, top=0.95, wspace=0.10)
    else:
        fig.subplots_adjust(left=0.035, right=0.975, bottom=0.16, top=0.95)
    
    output_file = output_dir / f'{timestamp}_residual_plot_chi2.pdf'
    plt.savefig(output_file, dpi=150)
    plt.close()
    
    print(f"\nResidual plot saved to: {output_file}")


def save_results(output_dir, true_loglkls, surr1_loglkls, surr2_loglkls, 
                 errors1, errors2, samples, run1_name, run2_name):
    """Save comparison results to file."""
    output_dir = Path(output_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    results = {
        'run1_name': run1_name,
        'run2_name': run2_name,
        'true_loglkls': true_loglkls,
        'surr1_loglkls': surr1_loglkls,
        'surr2_loglkls': surr2_loglkls,
        'errors1': {k: v for k, v in errors1.items() if k not in ['abs_errors', 'relative_errors']},
        'errors2': {k: v for k, v in errors2.items() if k not in ['abs_errors', 'relative_errors']},
        'samples': samples,
        'timestamp': timestamp
    }
    
    results_file = output_dir / f'{timestamp}_comparison_results.pkl'
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to: {results_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare Gaussian likelihood models from two runs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python compare_gaussian_models.py \\
      results/20251216_170029_gaussian \\
      results/20251218_124900_gaussian_k23 \\
      -n 5000 -it 9
        """
    )
    
    parser.add_argument('run1', help='Path to first run directory')
    parser.add_argument('run2', help='Path to second run directory')
    parser.add_argument('-it', '--iteration', type=int, default=9,
                       help='Iteration to compare (default: 9)')
    parser.add_argument('-n', '--n-samples', type=int, default=2000,
                       help='Number of test samples (default: 2000)')
    parser.add_argument('-s', '--strategy', choices=['lhs', 'random', 'grid'], default='lhs',
                       help='Sampling strategy for test points (default: lhs)')
    parser.add_argument('-o', '--output', default=None,
                       help='Output directory (default: comparison_results)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--parallel', action='store_true',
                       help='Use MPI for parallel evaluation of true likelihood')
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = 'comparison_results'
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    run1_name = Path(args.run1).name
    run2_name = Path(args.run2).name
    
    print("="*80)
    print("GAUSSIAN LIKELIHOOD MODEL COMPARISON")
    print("="*80)
    print(f"\nRun 1: {args.run1} ({run1_name})")
    print(f"Run 2: {args.run2} ({run2_name})")
    print(f"Iteration: {args.iteration}")
    print(f"Test samples: {args.n_samples}")
    print(f"Sampling strategy: {args.strategy}")
    print(f"Output directory: {output_dir}")
    print()

    print("Loading models...")
    surr1, true_likelihood1, cfg1 = load_model_and_likelihood(args.run1, args.iteration)
    surr2, true_likelihood2, cfg2 = load_model_and_likelihood(args.run2, args.iteration)
    
    print(f"  Model 1: {cfg1.wrapper} - kappa_sigma={getattr(cfg1, 'kappa_sigma', 'N/A')}")
    print(f"  Model 2: {cfg2.wrapper} - kappa_sigma={getattr(cfg2, 'kappa_sigma', 'N/A')}")
    
    true_likelihood = true_likelihood1
    param_names = true_likelihood.varying_param_names
    print(f"  Parameters: {param_names}")
    print()
    
    print(f"Generating {args.n_samples} test points using {args.strategy} strategy...")
    samples = generate_test_points(true_likelihood, args.n_samples, args.strategy, args.seed)
    print(f"  Generated {len(samples)} test points")
    print()
    
    print("\nLoading chain samples from gaussian_planck...")
    try:
        chain_samples = load_chain_samples('gaussian_planck', max_samples=args.n_samples, seed=args.seed)
        print(f"  Loaded {len(chain_samples)} chain samples")
    except Exception as e:
        print(f"  Warning: Could not load chain samples: {e}")
        chain_samples = None
    print()
    
    print("Evaluating models on random prior samples...")
    true_loglkls, surr1_loglkls, surr2_loglkls = evaluate_models(
        samples, true_likelihood, surr1, surr2, args.parallel
    )
    print("  Evaluation complete!")
    print()
    
    if chain_samples is not None:
        print("Evaluating models on chain samples...")
        true_loglkls_chain, surr1_loglkls_chain, surr2_loglkls_chain = evaluate_models(
            chain_samples, true_likelihood, surr1, surr2, args.parallel
        )
        print("  Evaluation complete!")
        print()
    else:
        true_loglkls_chain = None
        surr1_loglkls_chain = None
        surr2_loglkls_chain = None
    
    print("Computing error metrics...")
    errors1 = compute_errors(true_loglkls, surr1_loglkls)
    errors2 = compute_errors(true_loglkls, surr2_loglkls)
    
    print_comparison_table(errors1, errors2, run1_name, run2_name)
    
    print("\nCreating comparison plots...")
    kappa_sigma1 = getattr(cfg1, 'kappa_sigma', None)
    kappa_sigma2 = getattr(cfg2, 'kappa_sigma', None)
    create_comparison_plots(
        true_loglkls, surr1_loglkls, surr2_loglkls,
        errors1, errors2, run1_name, run2_name, output_dir,
        kappa_sigma1=kappa_sigma1, kappa_sigma2=kappa_sigma2,
        true_loglkls_chain=true_loglkls_chain,
        surr1_loglkls_chain=surr1_loglkls_chain,
        surr2_loglkls_chain=surr2_loglkls_chain
    )
    
    save_results(
        output_dir, true_loglkls, surr1_loglkls, surr2_loglkls,
        errors1, errors2, samples, run1_name, run2_name
    )
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
