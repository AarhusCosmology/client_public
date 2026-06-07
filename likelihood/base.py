from abc import ABC, abstractmethod

class BaseLikelihood(ABC):

    @abstractmethod
    def get_param_names(self):
        pass

    @abstractmethod
    def get_param_labels(self):
        pass

    @abstractmethod
    def get_prior_bounds(self):
        pass

    @abstractmethod
    def restrict_prior_bounds(self, n_sigma):
        pass

    @abstractmethod
    def loglkl(self, x):
        """Log-likelihood at ``x``, an ordered array of parameter values
        matching ``get_param_names()``."""
        pass

    @abstractmethod
    def logprior(self, x):
        pass

    @abstractmethod
    def logpost(self, x):
        pass

    def get_param_scales(self):
        """Per-parameter factors mapping stored-chain units to physical units.
        Unity by default; wrappers whose sampler stores parameters in rescaled
        units (e.g. MontePython) override this."""
        return [1.0] * len(self.get_param_names())


def build_likelihood(wrapper, input_file):
    if wrapper == 'montepython':
        from .montepython_wrapper import MontePythonLikelihood
        return MontePythonLikelihood(input_file=input_file)
    if wrapper == 'cobaya':
        from .cobaya_wrapper import CobayaLikelihood
        return CobayaLikelihood(yaml_file=input_file)
    raise ValueError(f"Unknown likelihood wrapper: {wrapper!r}")
