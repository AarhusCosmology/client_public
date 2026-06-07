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
        pass

    @abstractmethod
    def logprior(self, x):
        pass

    @abstractmethod
    def logpost(self, x):
        pass

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
