# metrics/convergence.py

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd


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
        return self._metric(chain.mean(axis=0), chain.std(axis=0),
                            prev_summary['mean'], prev_summary['std'])

    def _metric(self, mu_c, sig_c, mu_p, sig_p):
        sigma_bar = np.where(0.5 * (sig_c + sig_p) > 1e-10, 0.5 * (sig_c + sig_p), 1.0)
        return float((np.maximum(np.abs(mu_c - mu_p), np.abs(sig_c - sig_p)) / sigma_bar).max())


def build_convergence_metric(name):
    registry = {
        'marginal_r_minus_one': MarginalRMinusOne,
    }
    if name not in registry:
        raise ValueError(f"Unknown convergence metric: '{name}'. Available: {list(registry)}")
    return registry[name]()


def save_chain_summary(convergence_stats_dir, iteration, summary):
    path = Path(convergence_stats_dir) / f'chain_summary_it_{iteration}.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'mean': summary['mean'], 'std': summary['std']}).to_csv(path, index=False)
    return path


def load_chain_summary(convergence_stats_dir, iteration):
    path = Path(convergence_stats_dir) / f'chain_summary_it_{iteration}.csv'
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return {'mean': df['mean'].to_numpy(), 'std': df['std'].to_numpy()}, path
