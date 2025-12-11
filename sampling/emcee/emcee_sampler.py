# sampling/emcee/emcee_sampler.py

import sys
import time
import numpy as np
import emcee
import h5py

def _init_emcee_h5(path, nwalkers, ndim, group="mcmc", dtype=np.float64):
    with h5py.File(path, "a") as f:
        if group in f:
            del f[group]
        g = f.create_group(group)
        g.create_dataset(
            "chain",
            shape=(0, nwalkers, ndim),
            maxshape=(None, nwalkers, ndim),
            dtype=dtype,
            chunks=(1, nwalkers, ndim),
        )
        g.create_dataset(
            "log_prob",
            shape=(0, nwalkers),
            maxshape=(None, nwalkers),
            dtype=dtype,
            chunks=(1, nwalkers),
        )
        g.create_dataset("accepted", shape=(nwalkers,), dtype=dtype)
        g["accepted"][...] = 0.0
        g.attrs["iteration"] = 0
        g.attrs["ndim"] = int(ndim)
        g.attrs["nwalkers"] = int(nwalkers)
        g.attrs["version"] = str(getattr(emcee, "__version__", "unknown"))
        g.attrs["has_blobs"] = np.bool_(False)
        g.attrs["thin"] = 1


def _append_emcee_h5(path, chain_chunk, logp_chunk, acc_frac_per_walker=None, group="mcmc"):
    if chain_chunk.size == 0:
        return
    
    steps = int(chain_chunk.shape[0])
    with h5py.File(path, "a") as f:
        g = f[group]
        d_chain = g["chain"]
        d_logp = g["log_prob"]

        old = int(d_chain.shape[0])
        d_chain.resize(old + steps, axis=0)
        d_logp.resize(old + steps, axis=0)

        d_chain[old:old + steps] = chain_chunk
        d_logp[old:old + steps] = logp_chunk

        new_iter = old + steps
        g.attrs["iteration"] = int(new_iter)

        if acc_frac_per_walker is not None:
            g["accepted"][...] = np.asarray(acc_frac_per_walker, dtype=np.float64) * new_iter

        f.flush()


def _initialize_positions(posterior, n_walkers):
    n_params = len(posterior.varying_param_names)
    prior_bounds = posterior.get_prior_bounds()
    
    initial_positions = np.zeros((n_walkers, n_params))
    for i, param_name in enumerate(posterior.varying_param_names):
        bounds = prior_bounds[param_name]
        if len(bounds) != 2 or bounds[0] >= bounds[1]:
            raise ValueError(f"Invalid prior bounds for parameter {param_name}: {bounds}")
        initial_positions[:, i] = np.random.uniform(bounds[0], bounds[1], size=n_walkers)
    
    return initial_positions


def _create_tempered_logpost(posterior, temperature):
    def tempered_logpost(theta):
        if theta.ndim == 1:
            position = {name: float(theta[i]) for i, name in enumerate(posterior.varying_param_names)}
            logprior = posterior.logprior(position)
            if not np.isfinite(logprior):
                return -np.inf
            ll = posterior._loglkl(position)
            return ll / temperature + logprior
        else:
            loglkls = posterior.predict(theta)
            logpriors = posterior.logprior_array(theta)
            return loglkls / temperature + logpriors
    
    return tempered_logpost


def _run_burn_in(sampler, initial_positions, burn_in_steps, chain_path):
    if burn_in_steps <= 0:
        return initial_positions
    
    state = sampler.run_mcmc(initial_positions, burn_in_steps, progress=True)
    pos = state.coords
    
    if chain_path:
        start = sampler.iteration - burn_in_steps
        chain_chunk = sampler.get_chain()[start:sampler.iteration]
        logp_chunk = sampler.get_log_prob()[start:sampler.iteration]
        _append_emcee_h5(
            chain_path,
            chain_chunk,
            logp_chunk,
            acc_frac_per_walker=sampler.acceptance_fraction,
        )
    
    return pos


def _run_fixed_steps(sampler, pos, n_steps, chunk_size, chain_path):
    total_steps = 0
    last_written = sampler.iteration
    
    while total_steps < n_steps:
        steps_to_run = min(chunk_size, n_steps - total_steps)
        state = sampler.run_mcmc(pos, steps_to_run, progress=True)
        pos = state.coords
        total_steps += steps_to_run

        if chain_path:
            chain_chunk = sampler.get_chain()[last_written:sampler.iteration]
            logp_chunk = sampler.get_log_prob()[last_written:sampler.iteration]
            _append_emcee_h5(
                chain_path,
                chain_chunk,
                logp_chunk,
                acc_frac_per_walker=sampler.acceptance_fraction,
            )
            last_written = sampler.iteration

        acc_frac = float(np.mean(sampler.acceptance_fraction))
        sys.stdout.write(f"Steps {total_steps}/{n_steps}, Acceptance rate: {acc_frac:.2f}\n")
        sys.stdout.flush()
    
    return total_steps


def _run_adaptive_steps(sampler, pos, cfg, n_params, chain_path):
    total_steps = 0
    old_tau = np.inf * np.ones(n_params)
    last_written = sampler.iteration

    while total_steps < cfg.max_steps:
        state = sampler.run_mcmc(pos, cfg.chunk_size, progress=True)
        pos = state.coords
        total_steps += cfg.chunk_size

        if chain_path:
            chain_chunk = sampler.get_chain()[last_written:sampler.iteration]
            logp_chunk = sampler.get_log_prob()[last_written:sampler.iteration]
            _append_emcee_h5(
                chain_path,
                chain_chunk,
                logp_chunk,
                acc_frac_per_walker=sampler.acceptance_fraction,
            )
            last_written = sampler.iteration

        try:
            tau = sampler.get_autocorr_time(thin=cfg.ac_thin, tol=0)
        except Exception:
            print("Not enough samples to compute autocorrelation time, continuing sampling...")
            continue

        max_tau = float(np.max(tau))
        step_ratio = total_steps / (cfg.ess_target * max_tau)
        max_dta = float(np.max(np.abs((tau - old_tau) / old_tau))) if np.all(np.isfinite(old_tau)) else np.nan
        acc_frac = float(np.mean(sampler.acceptance_fraction))

        status = (
            f"Steps {total_steps}\n"
            f"max(τ): {max_tau:.2f}\n"
            f"Steps/{cfg.ess_target}max(τ): {step_ratio:.2f}\n"
            f"max(Δτ): {max_dta:.2f}\n"
            f"Acceptance rate: {acc_frac:.2f}\n"
        )
        sys.stdout.write(status)
        sys.stdout.flush()

        if total_steps < cfg.ess_target * max_tau:
            old_tau = tau.copy()
            continue

        if np.allclose(tau, old_tau, rtol=cfg.delta_tau_tol):
            break

        old_tau = tau.copy()
    
    return total_steps


def _collect_metrics(sampler, cfg, total_steps, n_params, n_steps, start_time):
    sampling_time = time.time() - start_time
    final_acceptance_rate = float(np.mean(sampler.acceptance_fraction))
    final_max_tau = None
    converged = True
    
    if n_steps is None:
        try:
            tau = sampler.get_autocorr_time(thin=cfg.ac_thin, tol=0)
            final_max_tau = float(np.max(tau))
            converged = total_steps < cfg.max_steps
        except Exception:
            converged = False
    
    return {
        'steps_to_convergence': total_steps,
        'acceptance_rate': final_acceptance_rate,
        'sampling_time': sampling_time,
        'final_max_tau': final_max_tau,
        'converged': converged
    }


def run_emcee_sampler(
    cfg,
    posterior,
    chain_path=None,
    vectorize=True,
    flat=True,
    temper=True,
    n_steps=None,
    return_metrics=False,
    return_sampler=False
):
    start_time = time.time() if return_metrics else None
    temperature = cfg.temperature if temper else 1.0
    n_params = len(posterior.varying_param_names)
    
    initial_positions = _initialize_positions(posterior, cfg.n_walkers)
    tempered_logpost = _create_tempered_logpost(posterior, temperature)
    
    sampler = emcee.EnsembleSampler(
        nwalkers=cfg.n_walkers,
        ndim=n_params,
        log_prob_fn=tempered_logpost,
        vectorize=vectorize
    )

    if chain_path:
        _init_emcee_h5(chain_path, cfg.n_walkers, n_params)

    pos = _run_burn_in(sampler, initial_positions, cfg.burn_in, chain_path)

    if n_steps is not None:
        total_steps = _run_fixed_steps(sampler, pos, n_steps, cfg.chunk_size, chain_path)
    else:
        total_steps = _run_adaptive_steps(sampler, pos, cfg, n_params, chain_path)

    samples = sampler.get_chain(discard=cfg.burn_in, flat=flat)
    tempered_loglkl = sampler.get_log_prob(discard=cfg.burn_in, flat=flat)
    
    if return_metrics and return_sampler:
        sampling_metrics = _collect_metrics(sampler, cfg, total_steps, n_params, n_steps, start_time)
        return samples, tempered_loglkl, sampling_metrics, sampler
    elif return_metrics:
        sampling_metrics = _collect_metrics(sampler, cfg, total_steps, n_params, n_steps, start_time)
        return samples, tempered_loglkl, sampling_metrics
    elif return_sampler:
        return samples, tempered_loglkl, sampler
    
    return samples, tempered_loglkl
