# metrics/convergence.py

import numpy as np
import pandas as pd
from pathlib import Path


def _summarise(chain):
    """Compute per-parameter marginal mean and std from a (steps, walkers, ndim) chain."""
    flat = chain.reshape(-1, chain.shape[-1])
    return {'mean': flat.mean(axis=0), 'std': flat.std(axis=0), 'cov': np.cov(flat, rowvar=False)}


def _marginal_r_minus_one(current_summary, prev_summary):
    """Max over parameters of max(|Δμ|, |Δσ|) / σ̄."""
    mu_c,  sig_c  = current_summary['mean'], current_summary['std']
    mu_p,  sig_p  = prev_summary['mean'],    prev_summary['std']
    sigma_bar = 0.5 * (sig_c + sig_p)
    sigma_bar = np.where(sigma_bar > 1e-10, sigma_bar, 1.0)
    return float(np.maximum(np.abs(mu_c - mu_p), np.abs(sig_c - sig_p)).max() / sigma_bar.min())


def _multivariate_r_minus_one(current_summary, prev_summary):
    """Eigenvalue-based multivariate Gelman-Rubin R-1."""
    mean_i,  cov_i  = current_summary['mean'], current_summary['cov']
    mean_im1, cov_im1 = prev_summary['mean'],  prev_summary['cov']
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
        L_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals))
    M = L_inv @ corr_means @ L_inv.T
    return float(np.max(np.linalg.eigvalsh(M)))


def _save_summary(stats_dir, iteration, summary):
    path = Path(stats_dir)
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'mean': summary['mean'], 'std': summary['std']}).to_csv(
        path / f'chain_summary_it_{iteration}.csv', index=False
    )
    np.save(path / f'chain_cov_it_{iteration}.npy', summary['cov'])


def _load_summary(stats_dir, iteration):
    csv_path = Path(stats_dir) / f'chain_summary_it_{iteration}.csv'
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    summary = {'mean': df['mean'].to_numpy(), 'std': df['std'].to_numpy()}
    cov_path = Path(stats_dir) / f'chain_cov_it_{iteration}.npy'
    summary['cov'] = np.load(cov_path) if cov_path.exists() else None
    return summary


def check_convergence(cfg, iteration, chain):
    """Summarise `chain`, save to CSV/npy, and compute both R-1 metrics against iteration-1.

    Returns (converged, r_minus_one, r_minus_one_old).
    Both R-1 values are None when iteration < 1 or previous summary is missing.
    """
    summary = _summarise(chain)
    _save_summary(cfg.convergence_stats_dir, iteration, summary)

    if iteration < 1:
        return False, None, None

    prev_summary = _load_summary(cfg.convergence_stats_dir, iteration - 1)
    if prev_summary is None:
        return False, None, None

    r_minus_one     = _marginal_r_minus_one(summary, prev_summary)
    r_minus_one_old = _multivariate_r_minus_one(summary, prev_summary) if prev_summary['cov'] is not None else None
    converged = r_minus_one < cfg.convergence_threshold
    return converged, r_minus_one, r_minus_one_old
