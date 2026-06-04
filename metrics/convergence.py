# metrics/convergence.py

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from pathlib import Path


class BaseConvergenceMetric(ABC):
    @abstractmethod
    def summarise(self, chain):
        """chain: (n_samples, ndim) flat array. Returns a summary dict."""
        pass

    @abstractmethod
    def compute_from_summary(self, chain, prev_summary):
        """chain: (n_samples, ndim) flat array. Returns scalar metric value."""
        pass


class MarginalRMinusOne(BaseConvergenceMetric):
    def summarise(self, chain):
        chain = np.asarray(chain)
        return {'mean': chain.mean(axis=0), 'std': chain.std(axis=0)}

    def compute_from_summary(self, chain, prev_summary):
        chain = np.asarray(chain)
        mu_c,  sig_c = chain.mean(axis=0), chain.std(axis=0)
        mu_p,  sig_p = prev_summary['mean'], prev_summary['std']
        sigma_bar = np.where(0.5 * (sig_c + sig_p) > 1e-10, 0.5 * (sig_c + sig_p), 1.0)
        return float((np.maximum(np.abs(mu_c - mu_p), np.abs(sig_c - sig_p)) / sigma_bar).max())


def build_convergence_metric(name):
    registry = {
        'marginal_r_minus_one': MarginalRMinusOne,
    }
    if name not in registry:
        raise ValueError(f"Unknown convergence metric: '{name}'. Available: {list(registry)}")
    return registry[name]()


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


def check_convergence(metric, cfg, iteration, chain):
    """Summarise flat `chain`, save to CSV, and compute metric against iteration-1.

    Returns (converged, value). value is None when iteration < 1 or the previous
    summary has not been saved yet.
    """
    summary = metric.summarise(chain)
    _save_summary(cfg.convergence_stats_dir, iteration, summary)

    if iteration < 1:
        return False, None

    prev_summary = _load_summary(cfg.convergence_stats_dir, iteration - 1)
    if prev_summary is None:
        return False, None

    value = metric.compute_from_summary(chain, prev_summary)
    converged = value < cfg.convergence_threshold
    return converged, value
