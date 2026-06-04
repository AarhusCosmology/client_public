# metrics/convergence.py

import os
import numpy as np
import pickle
from pathlib import Path


def compute_chain_statistics(chain, n_samples=100000):
    n_steps, _, n_params = chain.shape
    
    chain = chain[n_steps // 2:]
    
    flat_chain = chain.reshape(-1, n_params)
    n_total_samples = flat_chain.shape[0]
    
    n_subsample = min(n_samples, n_total_samples)
    indices = np.random.choice(n_total_samples, size=n_subsample, replace=False)
    subsampled_chain = flat_chain[indices]
    
    mean = np.mean(subsampled_chain, axis=0)
    cov = np.cov(subsampled_chain, rowvar=False)
    
    return mean, cov


def compute_gelman_rubin_from_stats(mean_i, cov_i, mean_im1, cov_im1):
    n_params = mean_i.shape[0]
    
    means = np.array([mean_im1, mean_i])
    
    W = (cov_im1 + cov_i) / 2
    B = np.atleast_2d(np.cov(means, rowvar=False))

    d = np.sqrt(np.diag(B))
    d = np.where(d > 1e-10, d, 1.0)
    
    corr_means = (B / d).T / d
    norm_W = (W / d).T / d
    norm_W += 1e-8 * np.eye(n_params)
    
    try:
        L = np.linalg.cholesky(norm_W)
        L_inv = np.linalg.inv(L)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(norm_W)
        eigvals = np.maximum(eigvals, 1e-8)
        L = eigvecs @ np.diag(np.sqrt(eigvals))
        L_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals))
    
    M = L_inv @ corr_means @ L_inv.T
    eigenvalues = np.linalg.eigvalsh(M)
    
    return float(np.max(eigenvalues))


def save_chain_statistics(stats_dir, iteration, mean, cov):
    os.makedirs(stats_dir, exist_ok=True)
    stats_path = os.path.join(stats_dir, f'chain_stats_it_{iteration}.pkl')
    
    with open(stats_path, 'wb') as f:
        pickle.dump({'mean': mean, 'cov': cov}, f)


def load_chain_statistics(stats_dir, iteration):
    stats_path = os.path.join(stats_dir, f'chain_stats_it_{iteration}.pkl')
    
    if not os.path.exists(stats_path):
        return None, None
    
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    
    return stats['mean'], stats['cov']


def compute_and_save_statistics(cfg, iteration, chain):
    n_samples = getattr(cfg, 'convergence_n_samples', 100000)
    mean, cov = compute_chain_statistics(chain, n_samples=n_samples)
    save_chain_statistics(cfg.convergence_stats_dir, iteration, mean, cov)
    return mean, cov


def check_convergence(cfg, iteration, chain):
    """Save statistics for `iteration` and compute R-1 against iteration-1.

    Returns (converged, r_minus_one). r_minus_one is None when iteration < 1
    or when previous stats have not been saved yet.
    """
    mean_i, cov_i = compute_and_save_statistics(cfg, iteration, chain)

    if iteration < 1:
        return False, None

    mean_im1, cov_im1 = load_chain_statistics(cfg.convergence_stats_dir, iteration - 1)
    if mean_im1 is None:
        return False, None

    r_minus_one = compute_gelman_rubin_from_stats(mean_i, cov_i, mean_im1, cov_im1)
    converged = r_minus_one < cfg.convergence_threshold
    return converged, r_minus_one
