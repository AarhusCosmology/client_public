from abc import ABC, abstractmethod

class BaseLikelihood(ABC):

    @abstractmethod
    def loglkl(self, position):
        pass

    @abstractmethod
    def logprior(self, position):
        pass

    @abstractmethod
    def logpost(self, position):
        pass


def build_likelihood(wrapper, input_file):
    if wrapper == 'montepython':
        from .montepython_wrapper import MontePythonLikelihood
        return MontePythonLikelihood(input_file=input_file)
    if wrapper == 'cobaya':
        from .cobaya_wrapper import CobayaLikelihood
        return CobayaLikelihood(yaml_file=input_file)
    raise ValueError(f"Unknown likelihood wrapper: {wrapper!r}")
