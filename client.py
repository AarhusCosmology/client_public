# client.py

import os
import time
import argparse
import numpy as np

from config.config_loader import load_config_cli
from config.run_manager import write_run_log, append_convergence_info
from metrics.metrics_tracker import MetricsTracker
from metrics.convergence import check_convergence
from likelihood.base import build_likelihood
from sampling.initial_sampler import generate_samples
from utils.mpi_utils import (
    is_mpi_available,
    is_master,
    get_size,
    print_master,
    barrier,
    get_communicator,
    parallel_evaluate_likelihood,
    bcast_array,
)

def parse_arguments():
    parser = argparse.ArgumentParser(description='CLiENT: Cosmological Likelihood Emulator using Neural networks with Tensorflow')
    
    parser.add_argument('input_or_dir', help='Input YAML file (new run) or run directory (continue)')
    parser.add_argument('-n', '--name', help='Run name/tag for organization (new runs only)')
    parser.add_argument('-o', '--output', default='results', help='Base directory for results (new runs only)')
    parser.add_argument('-r', '--retrain', action='store_true', help='Retrain model for starting iteration (continue only)')
    parser.add_argument('-s', '--start-it', type=int, help='Starting iteration (continue only, auto-detected if not specified)')
    parser.add_argument('-i', '--n-it', type=int, help='Number of new training iterations to produce (overrides convergence criterion)')
    
    args = parser.parse_args()
    
    from pathlib import Path
    path = Path(args.input_or_dir)
    
    if path.is_dir() and (path / 'training_data').exists():
        args.mode = 'continue'
        args.run_dir = args.input_or_dir
    else:
        args.mode = 'default'
        args.input = args.input_or_dir
    
    return args


def initialize_configuration(args, using_mpi):
    if is_master():
        cfg = load_config_cli(args)
        metrics_tracker = MetricsTracker(cfg.run_dir, start_iteration=cfg.start_it)
        append_log = cfg.run_mode != 'default'
        config_name_for_log = args.input if args.mode == 'default' else args.run_dir
        write_run_log(cfg, config_name_for_log, append=append_log)
        
        print(f"Run: {cfg.run_id}")
        print(f"Results directory: {cfg.run_dir}")
        print(f"Run mode: {cfg.run_mode}")
        
        if cfg.n_it is not None:
            end_it = cfg.start_it + cfg.n_it - 1
            if cfg.run_mode in ['skip_retrain_continue', 'retrain_continue']:
                end_it += 1
            print(f"Iterations: {cfg.start_it} to {end_it}")
        else:
            print(f"Running with convergence criterion: R-1 < {cfg.convergence_threshold}")
            print(f"Maximum iterations: {cfg.max_iterations}")
        print()
    else:
        cfg = None
        metrics_tracker = None
    
    if using_mpi:
        cfg = get_communicator().bcast(cfg, root=0)
    
    return cfg, metrics_tracker


def initialize_likelihood(cfg):
    print_master("Initializing likelihood...")
    likelihood = build_likelihood(cfg.wrapper, cfg.param)
    n_sigma = getattr(cfg, 'n_sigma', None)
    if n_sigma is not None:
        likelihood.restrict_prior_bounds(n_sigma=n_sigma)
    return likelihood


def generate_initial_samples(cfg, likelihood, using_mpi):
    if is_master():
        samples = generate_samples(
            likelihood=likelihood,
            n_samples=cfg.n_samples,
            strategy='lhs'
        )
    else:
        samples = None
    
    if using_mpi:
        samples = bcast_array(samples)
    
    return samples


def evaluate_initial_samples(cfg, likelihood, samples, using_mpi):
    print_master(f"Evaluating {cfg.n_samples} initial samples...")
    initial_start = time.time()
    
    param_names = likelihood.varying_param_names
    likelihood_func = lambda x: likelihood.loglkl({name: float(x[j]) for j, name in enumerate(param_names)})
    loglkls = parallel_evaluate_likelihood(samples, likelihood_func, description="initial samples")
    
    if is_master():
        elapsed = time.time() - initial_start
        print_master(f"Initial sampling completed in {elapsed:.2f}s ({elapsed/cfg.n_samples:.2f}s per sample)\n")
        
        outlier_threshold = -1e20
        valid_mask = loglkls > outlier_threshold
        n_outliers = np.sum(~valid_mask)
        
        if n_outliers > 0:
            print(f"Warning: Filtered out {n_outliers}/{len(loglkls)} samples with log-likelihood < {outlier_threshold:.1e}")
            samples = samples[valid_mask]
            loglkls = loglkls[valid_mask]
            
            if len(samples) == 0:
                raise ValueError(f"All initial samples had log-likelihood < {outlier_threshold:.1e}. Check prior ranges and likelihood configuration.")
    else:
        samples = None
        loglkls = None
    
    return samples, loglkls


def load_training_history(cfg, start_it, load_training_data):
    if is_master():
        x_list, y_list = [], []
        for i in range(start_it + 1):
            x, y = load_training_data(os.path.join(cfg.training_data_dir, f'data_it_{i}.h5'))
            x_list.append(x)
            y_list.append(y)
        return np.vstack(x_list), np.concatenate(y_list)
    return None, None


def should_train_model(iteration_idx, cfg):
    if cfg.run_mode == 'skip_retrain_continue':
        return iteration_idx > 0
    return True


def train_iteration_model(cfg, likelihood, x_all, y_all, iteration, build_model_func, train_model_func):
    if not is_master():
        return None

    from training.losses import build_loss

    n_params = len(likelihood.varying_param_names)
    shuffle_idx = np.random.permutation(len(x_all))
    inputs  = x_all[shuffle_idx].astype(np.float32)
    targets = y_all[shuffle_idx].reshape(-1, 1).astype(np.float32)

    loss = build_loss(
        name=cfg.loss_func,
        kappa=cfg.kappa_sigma,
        n=n_params,
        y_global_max=float(targets.max()),
    )

    model = build_model_func(x_train=inputs, n_layers=cfg.n_layers,
                             n_neurons=cfg.n_neurons, activation=cfg.act_func)

    history, training_metrics = train_model_func(
        model=model,
        inputs=inputs,
        targets=targets,
        loss=loss,
        learning_rate=cfg.learning_rate,
        n_epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        validation_split=cfg.validation_split,
        patience=cfg.patience,
        return_metrics=True,
    )

    model_path = os.path.join(cfg.trained_models_dir, f'trained_model_it_{iteration}.keras')
    model.save(model_path)

    history_path = os.path.join(cfg.training_history_dir, f'history_it_{iteration}.csv')
    from training.training import save_history
    save_history(history.history, history_path)

    return training_metrics


def run_sampling_step(cfg, likelihood, iteration, sampler=None, surrogate=None):
    if not is_master():
        return None, None, None, None, None, None

    from model.network import load_model
    from likelihood.surrogate import SurrogateLikelihood
    from sampling.sampler import build_sampler

    model_path = os.path.join(cfg.trained_models_dir, f'trained_model_it_{iteration}.keras')

    model = load_model(model_path)

    if surrogate is None:
        # First iteration: build surrogate and sampler from scratch.
        surrogate = SurrogateLikelihood(true_likelihood=likelihood, model=model)
        param_names = likelihood.varying_param_names
        ndim = len(param_names)
        prior_bounds = likelihood.get_prior_bounds()
        temperature = cfg.temperature

        logpost_fn = lambda positions: surrogate.logpost(positions) / temperature

        sampler = build_sampler(
            name=cfg.sampler,
            n_walkers=cfg.n_walkers,
            ndim=ndim,
            logpost_fn=logpost_fn,
        )
    else:
        # Subsequent iterations: update model weights in-place so the compiled
        # logpost_fn graph stays valid without retracing.
        surrogate.model.set_weights(model.get_weights())
        prior_bounds = likelihood.get_prior_bounds()

    ndim = len(likelihood.varying_param_names)
    initial_pos = np.random.uniform(
        low =[b[0] for b in prior_bounds.values()],
        high=[b[1] for b in prior_bounds.values()],
        size=(cfg.n_walkers, ndim),
    )

    start_time = time.time()
    sampler.run(
        initial_pos=initial_pos,
        max_steps=cfg.max_steps,
        chunk_size=cfg.chunk_size,
        target_ess=cfg.target_ess,
        tau_stability=cfg.tau_stability,
        iat_memory_mb=cfg.iat_memory_mb,
    )
    sampling_time = time.time() - start_time

    chain    = sampler.get_chain(discard=cfg.burn_in, flat=True)
    logposts = sampler.get_logpost(discard=cfg.burn_in, flat=True)

    if hasattr(chain, 'numpy'):
        chain    = chain.numpy()
        logposts = logposts.numpy()

    loglkls = logposts * cfg.temperature

    steps_done = sampler.get_chain().shape[0] + cfg.burn_in if hasattr(sampler, '_chain') and sampler._chain is not None else cfg.max_steps
    if hasattr(sampler, '_sampler'):
        steps_done = sampler._sampler.iteration

    sampling_metrics = {
        'steps_to_convergence': int(steps_done),
        'acceptance_rate': 0.0,
        'sampling_time': sampling_time,
    }

    full_chain = sampler.get_chain(discard=cfg.burn_in)
    if hasattr(full_chain, 'numpy'):
        full_chain = full_chain.numpy()

    return chain, loglkls, sampling_metrics, sampler, surrogate, full_chain


def run_resampling_step(cfg, likelihood, samples, loglkls, dataset, surrogate, iteration):
    """Augment the training dataset with new points selected from the MCMC chain.

    Master process runs TrainingDataset.augment(); non-master returns empty arrays.
    """
    if not is_master():
        ndim = len(likelihood.varying_param_names)
        return np.empty((0, ndim)), np.empty(0), {}

    print_master(f"Starting resampling for iteration {iteration}...")
    n_before = len(dataset.inputs)
    start = time.time()

    # loglkls from run_sampling_step are raw log-likelihoods (log_L);
    # TrainingDataset.augment expects tempered log-posteriors (log_L / T_MC).
    logposts_tempered = (loglkls / cfg.temperature).astype(np.float64)

    dataset.augment(
        chain=samples,
        logposts=logposts_tempered,
        surrogate=surrogate,
        n_augment=cfg.n_augment,
        sampling_temperature=cfg.temperature,
        pool_factor=cfg.pool_factor,
    )
    elapsed = time.time() - start

    n_added = len(dataset.inputs) - n_before
    x_new = dataset.inputs[n_before:].copy()
    y_new = dataset.targets.flatten()[n_before:].copy()

    print_master(f"Resampling completed in {elapsed:.2f}s")
    if n_added > 0:
        print_master(f"   Accepted {n_added} new samples ({elapsed/n_added:.2f}s per accepted sample)\n")
    else:
        print_master("")

    metrics = {
        'candidates_processed': min(cfg.pool_factor * cfg.n_augment, len(samples)),
        'accepted': n_added,
        'resampling_time': elapsed,
    }

    return x_new, y_new, metrics


def update_training_data(x_all, y_all, x_new, y_new, is_master_proc):
    if not is_master_proc and len(x_new) == 0:
        return x_all, y_all
    return np.concatenate([x_all, x_new], axis=0), np.concatenate([y_all, y_new], axis=0)


def main():
    using_mpi = is_mpi_available()
    
    if using_mpi:
        print_master(f"\nMPI enabled: {get_size()} processes\n")
    else:
        print_master("\nRunning in serial mode (no MPI)\n")
    
    args = parse_arguments()
    cfg, metrics_tracker = initialize_configuration(args, using_mpi)
    
    likelihood = initialize_likelihood(cfg)
    
    from model.network import build_model
    from training.training import train_model, save_training_data, load_training_data
    from training.dataset import TrainingDataset
    
    if cfg.run_mode == 'default':
        start_it = 0
        x_init = generate_initial_samples(cfg, likelihood, using_mpi)
        x_init, y_init = evaluate_initial_samples(cfg, likelihood, x_init, using_mpi)
        
        if is_master():
            save_training_data(x_init, y_init, os.path.join(cfg.training_data_dir, f'data_it_{start_it}.h5'))
            x_all, y_all = x_init, y_init
        else:
            x_all, y_all = None, None
    else:
        start_it = cfg.start_it
        x_all, y_all = load_training_history(cfg, start_it, load_training_data)
        
        if using_mpi:
            x_all = bcast_array(x_all)
            y_all = bcast_array(y_all)

    if cfg.n_it is not None:
        max_iterations = cfg.n_it
        use_convergence = False
        if cfg.run_mode in ['skip_retrain_continue', 'retrain_continue']:
            max_iterations += 1
    elif cfg.convergence_enabled:
        max_iterations = cfg.max_iterations
        use_convergence = True
    else:
        max_iterations = 1
        use_convergence = False
    
    if is_master():
        _dataset = TrainingDataset(
            inputs=x_all.astype(np.float32),
            targets=y_all.reshape(-1, 1).astype(np.float32),
            likelihood=likelihood,
            n_neighbors=cfg.n_neighbors,
            target_temperature=cfg.target_temperature,
        )
    else:
        _dataset = None

    final_iteration = start_it
    final_converged = False
    _sampler   = None
    _surrogate = None
    
    for i in range(max_iterations):
        iteration_start = time.time()
        iteration = start_it + i
        final_iteration = iteration
        
        if use_convergence:
            print_master(f"--- Iteration {iteration} (max: {start_it + max_iterations - 1}) ---")
        else:
            print_master(f"--- Iteration {iteration}/{start_it + max_iterations - 1} ---")
        
        if should_train_model(i, cfg):
            training_metrics = train_iteration_model(cfg, likelihood, x_all, y_all, iteration, build_model, train_model)
            if is_master():
                metrics_tracker.add_training_metrics(iteration=iteration, **training_metrics)

        barrier()

        samples, loglkls, sampling_metrics, _sampler, _surrogate, current_chain = run_sampling_step(
            cfg, likelihood, iteration, sampler=_sampler, surrogate=_surrogate
        )
        
        if is_master():
            metrics_tracker.add_sampling_metrics(iteration=iteration, **sampling_metrics)

        converged = False
        if is_master():
            from metrics.convergence import compute_and_save_statistics
            compute_and_save_statistics(cfg, iteration, current_chain)
            
            if iteration >= 1:
                converged, r_minus_one = check_convergence(cfg, iteration, current_chain=current_chain)
                
                if r_minus_one is not None:
                    print_master(f"\nGelman-Rubin R-1: {r_minus_one:.6f} (threshold: {cfg.convergence_threshold})")
                    metrics_tracker.add_convergence_metrics(iteration, r_minus_one, converged)
                    
                    if use_convergence and converged:
                        print_master(f"\nConvergence achieved at iteration {iteration}!\n")
                        final_converged = True
                        iteration_time = time.time() - iteration_start
                        metrics_tracker.add_iteration_metrics(iteration, iteration_time)
                        metrics_tracker.save_all_metrics()
                else:
                    print_master("\nR-1 not yet calculable (need at least 2 iterations)\n")
        
        if use_convergence:
            if using_mpi:
                converged = get_communicator().bcast(converged, root=0)
            
            if converged:
                break
        
        is_last_iteration = (i == max_iterations - 1)
        
        if not is_last_iteration:
            if using_mpi:
                samples = bcast_array(samples)
                loglkls = bcast_array(loglkls)
                if i == 0 and cfg.run_mode == 'default':
                    x_all = bcast_array(x_all)
                    y_all = bcast_array(y_all)
            
            x_new, y_new, resampling_metrics = run_resampling_step(
                cfg, likelihood, samples, loglkls, _dataset, _surrogate, iteration
            )
            
            if is_master():
                metrics_tracker.add_resampling_metrics(iteration=iteration, **resampling_metrics)
                save_training_data(x_new, y_new, os.path.join(cfg.training_data_dir, f'data_it_{iteration+1}.h5'))
            
            x_all, y_all = update_training_data(x_all, y_all, x_new, y_new, is_master())
        
        if is_master():
            iteration_time = time.time() - iteration_start
            metrics_tracker.add_iteration_metrics(iteration, iteration_time)
            metrics_tracker.save_progress_metrics(iteration)
    
    barrier()
    
    if is_master():
        metrics_tracker.save_all_metrics()
        
        append_convergence_info(cfg.run_dir, final_iteration, final_converged)
        
        print(f"Run completed: {cfg.run_id}")
        print(f"Results saved to: {cfg.run_dir}")
        print(f"Metrics saved to: {cfg.run_dir}/metrics.log")


if __name__ == "__main__":
    main()