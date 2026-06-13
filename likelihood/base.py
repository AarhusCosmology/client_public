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
        """
        Evaluate the backend's native log-probability at ``x``. For MontePython, this calls ``compute_lkl``. For Cobaya, this calls ``logposterior(..., return_derived=False)`` and returns its ``logpost`` value. Despite the method name, the returned value will include both log-likelihood and backend-defined log-prior contributions. Additional bounds imposed by ``restrict_prior_bounds`` are handled separately by ``logprior``. 
        """
        pass

    @abstractmethod
    def logprior(self, x):
        """
        Return the additional bounds-based log-prior at ``x``. Returns zero inside the current effective bounds and negative infinity outside them. This does not reproduce any prior terms already evaluated by the backend. 
        """
        pass

    def logpost(self, x):
        """
        Return the total log-posterior value at ``x``. The result is the sum of the backend's native log-probability, returned by ``loglkl``, and the additional bounds-based log-prior, returned by ``logprior``. Consequently, the result is negative infinity when ``x`` lies outside the current effective bounds.
        """
        return self.loglkl(x) + self.logprior(x)

    def get_param_scales(self):
        return [1.0] * len(self.get_param_names())

def build_likelihood(wrapper, input_file):
    if wrapper == 'montepython':
        from .montepython_wrapper import MontePythonLikelihood
        return MontePythonLikelihood(input_file=input_file)
    if wrapper == 'cobaya':
        from .cobaya_wrapper import CobayaLikelihood
        return CobayaLikelihood(yaml_file=input_file)
    raise ValueError(f"Unknown likelihood wrapper: {wrapper!r}")
