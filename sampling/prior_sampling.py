import numpy as np
from scipy.stats import qmc

def latin_hypercube(n_samples, named_bounds):
    names = list(named_bounds.keys())
    bounds = np.array([named_bounds[name] for name in names])

    sampler = qmc.LatinHypercube(d=len(names))
    unit_samples = sampler.random(n_samples)
    l_bounds, u_bounds = bounds[:, 0], bounds[:, 1]
    scaled_samples = qmc.scale(unit_samples, l_bounds, u_bounds)
    return [dict(zip(names, row)) for row in scaled_samples]