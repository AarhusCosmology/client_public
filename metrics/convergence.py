import numpy as np

from abc import ABC, abstractmethod
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

class GaussianPosteriorDrift(BaseConvergenceMetric):
    def __init__(self, relative_floor=1e-12):
        if relative_floor <= 0:
            raise ValueError("relative_floor must be positive")
        self.relative_floor = float(relative_floor)

    def _validate_chain(self, chain):
        chain = np.asarray(chain, dtype=np.float64)

        if chain.ndim != 2:
            raise ValueError(
                f"chain must have shape (n_samples, ndim), got {chain.shape}"
            )
        if chain.shape[0] < 2:
            raise ValueError("At least two samples are needed")
        if chain.shape[1] < 1:
            raise ValueError("chain must contain at least one parameter")
        if not np.all(np.isfinite(chain)):
            raise ValueError("chain contains NaN or infinite values")

        return chain

    def summarise(self, chain):
        chain = self._validate_chain(chain)

        return {
            "mean": chain.mean(axis=0),
            "cov": np.atleast_2d(np.cov(chain, rowvar=False)),
        }

    def compute_from_summary(self, chain, prev_summary):
        current = self.summarise(chain)

        prev_mean = np.asarray(prev_summary["mean"], dtype=np.float64)
        prev_cov = np.atleast_2d(
            np.asarray(prev_summary["cov"], dtype=np.float64)
        )

        if prev_mean.shape != current["mean"].shape:
            raise ValueError(
                "Current and previous means have incompatible shapes"
            )
        if prev_cov.shape != current["cov"].shape:
            raise ValueError(
                "Current and previous covariances have incompatible shapes"
            )
        if not np.all(np.isfinite(prev_mean)):
            raise ValueError("Previous mean contains NaN or infinite values")
        if not np.all(np.isfinite(prev_cov)):
            raise ValueError("Previous covariance contains NaN or infinite values")

        return self._metric(
            current["mean"],
            current["cov"],
            prev_mean,
            prev_cov,
        )

    def _regularized_eigh(self, cov):
        cov = np.asarray(cov, dtype=np.float64)

        # Remove small asymmetries introduced by floating-point operations.
        cov = 0.5 * (cov + cov.T)

        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Use the average marginal variance as the regularization scale.
        scale = float(np.trace(cov) / cov.shape[0])

        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(
                "Covariance has no positive scale; the convergence metric "
                "is undefined for a degenerate posterior"
            )

        floor = self.relative_floor * scale
        eigenvalues = np.maximum(eigenvalues, floor)

        return eigenvalues, eigenvectors
    
    def _regularize_cov(self, cov):
        eigenvalues, eigenvectors = self._regularized_eigh(cov)
        return (eigenvectors * eigenvalues[None, :]) @ eigenvectors.T

    def _inv_sqrt(self, cov):
        eigenvalues, eigenvectors = self._regularized_eigh(cov)

        return (
            eigenvectors
            * (1.0 / np.sqrt(eigenvalues))[None, :]
        ) @ eigenvectors.T

    def _metric(self, mu_c, cov_c, mu_p, cov_p):
        cov_c = self._regularize_cov(cov_c)
        cov_p = self._regularize_cov(cov_p)

        delta = mu_c - mu_p

        # Location drift.
        cov_bar = 0.5 * (cov_c + cov_p)
        whitened_delta = self._inv_sqrt(cov_bar) @ delta
        r_mean = float(np.linalg.norm(whitened_delta))

        # Shape drift.
        prev_inv_sqrt = self._inv_sqrt(cov_p)
        relative_cov = prev_inv_sqrt @ cov_c @ prev_inv_sqrt
        relative_cov = 0.5 * (relative_cov + relative_cov.T)

        eigenvalues = np.linalg.eigvalsh(relative_cov)

        # Both input covariances are already positive definite after
        # regularization, so only protect against tiny numerical negatives.
        eigenvalues = np.maximum(eigenvalues, np.finfo(np.float64).tiny)

        r_cov = 0.5 * float(np.max(np.abs(np.log(eigenvalues))))

        return max(r_mean, r_cov)


def build_convergence_metric(name):
    registry = {
        'gaussian_posterior_drift': GaussianPosteriorDrift
    }
    if name not in registry:
        raise ValueError(f"Unknown convergence metric: '{name}'. Available: {list(registry)}")
    return registry[name]()


def save_chain_summary(convergence_stats_dir, iteration, summary):
    path = Path(convergence_stats_dir) / f'chain_summary_it_{iteration}.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{k: np.asarray(v) for k, v in summary.items()})
    return path


def load_chain_summary(convergence_stats_dir, iteration):
    path = Path(convergence_stats_dir) / f'chain_summary_it_{iteration}.npz'
    if not path.exists():
        return None
    with np.load(path) as data:
        summary = {k: data[k] for k in data.files}
    return summary, path
