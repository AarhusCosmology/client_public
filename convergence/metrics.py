from abc import ABC, abstractmethod
import numpy as np


class BaseConvergenceMetric(ABC):
    @abstractmethod
    def compute(self, current, previous):
        pass

    @abstractmethod
    def summarise(self, chain):
        """Return a summary dict sufficient for a future compute_from_summary call."""
        pass

    @abstractmethod
    def compute_from_summary(self, current, summary):
        """Compute the metric given a current chain and a previously saved summary."""
        pass


class MarginalRMinusOne(BaseConvergenceMetric):
    def compute(self, current, previous):
        current  = np.asarray(current)
        previous = np.asarray(previous)
        mu_curr,  mu_prev  = current.mean(axis=0),  previous.mean(axis=0)
        sig_curr, sig_prev = current.std(axis=0),   previous.std(axis=0)
        return self._metric(mu_curr, sig_curr, mu_prev, sig_prev)

    def summarise(self, chain):
        chain = np.asarray(chain)
        return {'mean': chain.mean(axis=0), 'std': chain.std(axis=0)}

    def compute_from_summary(self, current, summary):
        current = np.asarray(current)
        mu_curr,  sig_curr  = current.mean(axis=0), current.std(axis=0)
        mu_prev,  sig_prev  = summary['mean'],       summary['std']
        return self._metric(mu_curr, sig_curr, mu_prev, sig_prev)

    def _metric(self, mu_curr, sig_curr, mu_prev, sig_prev):
        sigma_bar = 0.5 * (sig_curr + sig_prev)
        return float(np.maximum(
            np.abs(mu_curr  - mu_prev)  / sigma_bar,
            np.abs(sig_curr - sig_prev) / sigma_bar,
        ).max())


def build_convergence_metric(name):
    registry = {
        'marginal_r_minus_one': MarginalRMinusOne,
    }
    if name not in registry:
        raise ValueError(f"Unknown convergence metric: '{name}'. Available: {list(registry)}")
    return registry[name]()
