import numpy as np
from scipy.stats import qmc

def _scale_unit_samples_to_bounds(unit_samples, prior_bounds):
    samples = np.zeros_like(unit_samples)
    for i, (lower, upper) in enumerate(prior_bounds.values()):
        samples[:, i] = lower + unit_samples[:, i] * (upper - lower)
    return samples

def _lhc_samples(prior_bounds, n_samples):
    sampler = qmc.LatinHypercube(d=len(prior_bounds))
    return _scale_unit_samples_to_bounds(sampler.random(n=n_samples), prior_bounds)

def _uniform_samples(prior_bounds, n_samples):
    unit_samples = np.random.rand(n_samples, len(prior_bounds))
    return _scale_unit_samples_to_bounds(unit_samples, prior_bounds)

def sample_prior(likelihood, n_samples, strategy='lhs'):
    prior_bounds = likelihood.get_prior_bounds()
    if strategy == 'lhs':
        return _lhc_samples(prior_bounds, n_samples)
    if strategy == 'random':
        return _uniform_samples(prior_bounds, n_samples)
    raise ValueError(f"Unknown sampling strategy: {strategy}")
