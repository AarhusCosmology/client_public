import re

import numpy as np


def getdist_sample_inputs(chain, log_prob, sampler_name):
    """Convert sampler output into GetDist input arrays."""
    chain = np.asarray(chain)
    log_prob = np.asarray(log_prob)
    sampler_name = str(sampler_name).lower()

    if chain.ndim == 4:
        return _ensemble_chain_inputs(chain, log_prob)

    if chain.ndim == 3:
        _validate_log_prob_shape(chain, log_prob)
        if sampler_name == "aies":
            # Legacy AIES cache shape: (n_steps, n_walkers, ndim).
            samples = [chain[:, walker_idx, :] for walker_idx in range(chain.shape[1])]
            loglikes = [-log_prob[:, walker_idx] for walker_idx in range(log_prob.shape[1])]
            return samples, loglikes, len(samples) > 1

        # Non-ensemble samplers: (n_steps, n_chains, ndim).
        samples = [chain[:, chain_idx, :] for chain_idx in range(chain.shape[1])]
        loglikes = [-log_prob[:, chain_idx] for chain_idx in range(chain.shape[1])]
        return samples, loglikes, len(samples) > 1

    if chain.ndim == 2:
        # Already flattened output from older cache files.
        _validate_log_prob_shape(chain, log_prob)
        return chain, -log_prob, False

    raise ValueError(f"Unexpected chain shape {chain.shape}")


def getdist_names_for_params(param_names):
    return [re.sub(r"[\s*?]", "", name) for name in param_names]


def getdist_ranges_for_params(param_names, getdist_names, prior_bounds):
    return {
        getdist_name: prior_bounds[param_name]
        for param_name, getdist_name in zip(param_names, getdist_names)
    }


def select_plot_params(requested_params, param_names, getdist_names):
    if not requested_params:
        return getdist_names, list(range(len(param_names)))

    if len(requested_params) == 1 and "," in requested_params[0]:
        param_indices = [int(x) - 1 for x in requested_params[0].split(",")]
    else:
        param_indices = [param_names.index(param) for param in requested_params]

    return [getdist_names[i] for i in param_indices], param_indices


def _ensemble_chain_inputs(chain, log_prob):
    # AIES: (n_steps, n_chains, n_walkers, ndim).
    _validate_log_prob_shape(chain, log_prob)
    samples = [
        chain[:, chain_idx, walker_idx, :]
        for chain_idx in range(chain.shape[1])
        for walker_idx in range(chain.shape[2])
    ]
    loglikes = [
        -log_prob[:, chain_idx, walker_idx]
        for chain_idx in range(chain.shape[1])
        for walker_idx in range(chain.shape[2])
    ]
    return samples, loglikes, len(samples) > 1


def _validate_log_prob_shape(chain, log_prob):
    if log_prob.shape != chain.shape[:-1]:
        raise ValueError(
            f"log_prob shape {log_prob.shape} is incompatible with chain shape {chain.shape}"
        )
