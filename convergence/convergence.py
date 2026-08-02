from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseConvergenceMetric(ABC):
    @abstractmethod
    def summarize(self, chain):
        """chain: (n_steps, n_walkers, ndim) ensemble array. Returns a chain summary dict."""
        pass

    @abstractmethod
    def compute_from_summaries(self, current_chain_summary, previous_chain_summary):
        """summary dicts -> scalar metric value."""
        pass

    @staticmethod
    def save_chain_summary(convergence_stats_dir, iteration, chain_summary):
        path = Path(convergence_stats_dir) / f"chain_summary_it_{iteration}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **{k: np.asarray(v) for k, v in chain_summary.items()})
        return path

    @staticmethod
    def load_chain_summary(convergence_stats_dir, iteration):
        path = Path(convergence_stats_dir) / f"chain_summary_it_{iteration}.npz"
        if not path.exists():
            return None
        with np.load(path) as data:
            chain_summary = {k: data[k] for k in data.files}
        return chain_summary


class GaussianPosteriorDrift(BaseConvergenceMetric):
    def __init__(self, relative_floor=1e-12, name="gaussian_posterior_drift"):
        if relative_floor <= 0:
            raise ValueError("relative_floor must be positive")
        self.relative_floor = float(relative_floor)
        self.name = name

    def _validate_chain(self, chain):
        chain = np.asarray(chain)

        if chain.ndim != 3:
            raise ValueError(
                f"chain must have shape (n_steps, n_walkers, ndim), got {chain.shape}"
            )
        n_steps, n_walkers, ndim = chain.shape
        if n_steps < 1:
            raise ValueError("chain must contain at least one step")
        if n_walkers < 1:
            raise ValueError("chain must contain at least one walker or chain")
        if n_steps * n_walkers < 2:
            raise ValueError("At least two samples are needed")
        if ndim < 1:
            raise ValueError("chain must contain at least one parameter")

        return chain

    def summarize(self, chain):
        chain = self._validate_chain(chain)
        n_steps, n_walkers, ndim = chain.shape
        n_samples = n_steps * n_walkers

        sum_x = np.zeros(ndim, dtype=np.float64)
        for walker in range(n_walkers):
            walker_chain = chain[:, walker, :]
            if not np.all(np.isfinite(walker_chain)):
                raise ValueError("chain contains NaN or infinite values")
            sum_x += np.sum(walker_chain, axis=0, dtype=np.float64)

        mean = sum_x / n_samples
        scatter = np.zeros((ndim, ndim), dtype=np.float64)
        for walker in range(n_walkers):
            centered = np.array(chain[:, walker, :], dtype=np.float64, copy=True)
            centered -= mean
            scatter += centered.T @ centered

        cov = scatter / (n_samples - 1)
        return {
            "mean": mean,
            "cov": cov,
        }

    def compute_from_summaries(self, current_chain_summary, previous_chain_summary):
        current_mean = np.asarray(current_chain_summary["mean"], dtype=np.float64)
        current_cov = np.atleast_2d(
            np.asarray(current_chain_summary["cov"], dtype=np.float64)
        )

        prev_mean = np.asarray(previous_chain_summary["mean"], dtype=np.float64)
        prev_cov = np.atleast_2d(
            np.asarray(previous_chain_summary["cov"], dtype=np.float64)
        )

        if prev_mean.shape != current_mean.shape:
            raise ValueError("Current and previous means have incompatible shapes")
        if prev_cov.shape != current_cov.shape:
            raise ValueError(
                "Current and previous covariances have incompatible shapes"
            )
        if not np.all(np.isfinite(current_mean)):
            raise ValueError("Current mean contains NaN or infinite values")
        if not np.all(np.isfinite(current_cov)):
            raise ValueError("Current covariance contains NaN or infinite values")
        if not np.all(np.isfinite(prev_mean)):
            raise ValueError("Previous mean contains NaN or infinite values")
        if not np.all(np.isfinite(prev_cov)):
            raise ValueError("Previous covariance contains NaN or infinite values")

        return self._metric(
            current_mean,
            current_cov,
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

        return (eigenvectors * (1.0 / np.sqrt(eigenvalues))[None, :]) @ eigenvectors.T

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
    registry = {"gaussian_posterior_drift": GaussianPosteriorDrift}
    if name not in registry:
        raise ValueError(
            f"Unknown convergence metric: '{name}'. Available: {list(registry)}"
        )
    return registry[name](name=name)
