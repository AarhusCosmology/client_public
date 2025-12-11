# sampling/sampler.py

from sampling.emcee.emcee_sampler import run_emcee_sampler

def run_sampler(cfg, posterior, chain_path=None, vectorize=True, flat=True, temper=True, n_steps=None, return_metrics=False, return_sampler=False):
    if cfg.sampling_method == 'emcee':
        return run_emcee_sampler(cfg, posterior, chain_path, vectorize, flat, temper, n_steps, return_metrics, return_sampler)
    raise ValueError(f"Unknown sampling method: {cfg.sampling_method}")
