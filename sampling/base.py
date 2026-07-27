from abc import ABC, abstractmethod


class BaseSampler(ABC):
    @abstractmethod
    def run(self, n_steps, initial_positions=None, progress=True):
        pass

    @abstractmethod
    def chain(self, discard=0, thin=1):
        pass

    @abstractmethod
    def log_prob(self, discard=0, thin=1):
        pass

    @abstractmethod
    def acceptance_fraction(self):
        pass

    @abstractmethod
    def reset(self):
        pass


def build_sampler(name, n_walkers_or_chains, ndim, log_prob_fn, covmat=None, bounds=None):
    if name == "aies":
        from .aies import AIESampler
        return AIESampler(
            n_walkers=n_walkers_or_chains, ndim=ndim, log_prob_fn=log_prob_fn
        )
    if name == "nuts":
        from .nuts import NUTSampler
        return NUTSampler(
            n_chains=n_walkers_or_chains,
            ndim=ndim,
            log_prob_fn=log_prob_fn,
            covmat=covmat,
            bounds=bounds,
        )
    if name == "hmc":
        from .hmc import HMCSampler
        return HMCSampler(
            n_chains=n_walkers_or_chains,
            ndim=ndim,
            log_prob_fn=log_prob_fn,
            covmat=covmat,
            bounds=bounds,
        )
    raise ValueError(
        f"Unknown sampler name: {name}. Available samplers: ['aies', 'nuts', 'hmc']"
    )
