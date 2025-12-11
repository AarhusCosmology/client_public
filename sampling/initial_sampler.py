import numpy as np
from scipy.stats import qmc


def _scale_unit_samples_to_bounds(unit_samples, prior_bounds):
    param_names = list(prior_bounds.keys())
    samples = np.zeros_like(unit_samples)
    
    for i, param in enumerate(param_names):
        lower, upper = prior_bounds[param]
        samples[:, i] = lower + unit_samples[:, i] * (upper - lower)
    
    return samples

def generate_lhc_samples(prior_bounds, n_samples):
    n_params = len(prior_bounds)
    sampler = qmc.LatinHypercube(d=n_params)
    unit_samples = sampler.random(n=n_samples)
    return _scale_unit_samples_to_bounds(unit_samples, prior_bounds)

def generate_samples(likelihood, n_samples, strategy='lhs'):
    prior_bounds = likelihood.get_prior_bounds()
    
    if strategy == 'lhs':
        return generate_lhc_samples(prior_bounds, n_samples)
    raise ValueError(f"Unknown sampling strategy: {strategy}")
        