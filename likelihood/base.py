from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ParameterInfo:
    name: str
    label: str
    scale: float = 1.0
    lower: float | None = None
    upper: float | None = None
    center: float | None = None
    sigma: float | None = None


class BaseLikelihood(ABC):
    def __init__(self):
        self._params = self._extract_params()
        self._effective_bounds = None

    @abstractmethod
    def _extract_params(self):
        pass

    @abstractmethod
    def loglkl(self, x):
        """
        Evaluate the backend's native log-probability at ``x``. For MontePython, this calls ``compute_lkl``. For Cobaya, this calls ``logposterior(..., return_derived=False)`` and returns its ``logpost`` value. Despite the method name, the returned value will include both log-likelihood and backend-defined log-prior contributions. Additional bounds imposed by ``restrict_prior_bounds`` are handled separately by ``logprior``.
        """
        pass

    @property
    def param_names(self):
        return [param.name for param in self._params]

    @property
    def param_labels(self):
        return [param.label for param in self._params]

    @property
    def param_scales(self):
        return [param.scale for param in self._params]

    @property
    def param_sigmas(self):
        return [param.sigma for param in self._params]

    @property
    def ndim(self):
        return len(self._params)

    @property
    def prior_bounds(self):
        if self._effective_bounds is not None:
            return dict(self._effective_bounds)
        return {param.name: (param.lower, param.upper) for param in self._params}

    def restrict_prior_bounds(self, n_sigma):
        restricted_bounds = {}
        for param in self._params:
            lower = param.center - n_sigma * param.sigma
            upper = param.center + n_sigma * param.sigma
            if param.lower is not None:
                lower = max(lower, param.lower)
            if param.upper is not None:
                upper = min(upper, param.upper)
            restricted_bounds[param.name] = (lower, upper)
        self._effective_bounds = restricted_bounds

    def logprior(self, x):
        """
        Return the additional bounds-based log-prior at ``x``. Returns zero inside the current effective bounds and negative infinity outside them. This does not reproduce any prior terms already evaluated by the backend.
        """
        bounds = self.prior_bounds
        for value, param in zip(x, self._params):
            lower, upper = bounds[param.name]
            if lower is not None and value < lower:
                return -np.inf
            if upper is not None and value > upper:
                return -np.inf
        return 0.0

    def logpost(self, x):
        """
        Return the total log-posterior value at ``x``. The result is the sum of the backend's native log-probability, returned by ``loglkl``, and the additional bounds-based log-prior, returned by ``logprior``. Consequently, the result is negative infinity when ``x`` lies outside the current effective bounds.
        """
        lp = self.logprior(x)
        if not np.isfinite(lp):
            return -np.inf
        return self.loglkl(x) + lp


def build_likelihood(wrapper, input_path):
    if wrapper == "montepython":
        from .montepython import MontePythonLikelihood

        return MontePythonLikelihood(param_path=input_path)
    if wrapper == "cobaya":
        from .cobaya import CobayaLikelihood

        return CobayaLikelihood(yaml_path=input_path)
    raise ValueError(f"Unknown likelihood wrapper: {wrapper!r}")
