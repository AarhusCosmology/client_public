# metrics/convergence.py

import numpy as np
import pandas as pd
from pathlib import Path


def _summarise(chain):
    """Compute per-parameter marginal mean and std from a (steps, walkers, ndim) chain."""
    flat = chain.reshape(-1, chain.shape[-1])
    return {'mean': flat.mean(axis=0), 'std': flat.std(axis=0)}


def _marginal_r_minus_one(current_summary, prev_summary):
    """Max over parameters of max(|Δμ|, |Δσ|) / σ̄."""
    mu_c,  sig_c  = current_summary['mean'], current_summary['std']
    mu_p,  sig_p  = prev_summary['mean'],    prev_summary['std']
    sigma_bar = 0.5 * (sig_c + sig_p)
    sigma_bar = np.where(sigma_bar > 1e-10, sigma_bar, 1.0)
    return float(np.maximum(np.abs(mu_c - mu_p), np.abs(sig_c - sig_p)).max() / sigma_bar.min())


def _save_summary(stats_dir, iteration, summary):
    path = Path(stats_dir)
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'mean': summary['mean'], 'std': summary['std']}).to_csv(
        path / f'chain_summary_it_{iteration}.csv', index=False
    )


def _load_summary(stats_dir, iteration):
    path = Path(stats_dir) / f'chain_summary_it_{iteration}.csv'
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return {'mean': df['mean'].to_numpy(), 'std': df['std'].to_numpy()}


def check_convergence(cfg, iteration, chain):
    """Summarise `chain`, save to CSV, and compute marginal R-1 against iteration-1.

    Returns (converged, r_minus_one). r_minus_one is None when iteration < 1
    or when the previous summary has not been saved yet.
    """
    summary = _summarise(chain)
    _save_summary(cfg.convergence_stats_dir, iteration, summary)

    if iteration < 1:
        return False, None

    prev_summary = _load_summary(cfg.convergence_stats_dir, iteration - 1)
    if prev_summary is None:
        return False, None

    r_minus_one = _marginal_r_minus_one(summary, prev_summary)
    converged = r_minus_one < cfg.convergence_threshold
    return converged, r_minus_one
