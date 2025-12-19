# client.py

import os
import time
import argparse
import numpy as np

from config.config_loader import load_config_cli
from config.run_manager import write_run_log, append_convergence_info
from metrics.metrics_tracker import MetricsTracker
from metrics.convergence import check_convergence
from likelihood.montepython_wrapper import MontePythonLikelihood
from likelihood.cobaya_wrapper import CobayaLikelihood
from scaling.scaling import make_scalers, save_scalers
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
    
    if cfg.wrapper == 'montepython':
        likelihood = MontePythonLikelihood(
            param_file=cfg.param,
            conf_file=cfg.conf,
            montepython_path=cfg.path,
            silent=(not is_master())
        )
    elif cfg.wrapper == 'cobaya':
        likelihood = CobayaLikelihood(
            yaml_file=cfg.param,
            debug=False
        )
    else:
        raise ValueError(f"Unknown likelihood wrapper: {cfg.wrapper}")
    
    n_std = getattr(cfg, 'n_std', None)
    if n_std is not None:
        likelihood.restrict_prior(n_std=n_std)
    
    return likelihood


def generate_initial_samples(cfg, likelihood, using_mpi):
    if is_master():
        samples = generate_samples(
            likelihood=likelihood,
            n_samples=cfg.n_samples,
            strategy=cfg.s_strategy
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


def prepare_training_data(x_all, y_all, x_scaler, y_scaler):
    x_scaled = x_scaler.fit_transform(x_all)
    y_scaled = y_scaler.fit_transform(y_all.reshape(-1, 1))
    
    shuffle_indices = np.random.permutation(len(x_scaled))
    return x_scaled[shuffle_indices], y_scaled[shuffle_indices]


def train_iteration_model(cfg, likelihood, x_all, y_all, iteration, build_model_func, train_model_func):
    if not is_master():
        return None, None, None
    
    x_scaler, y_scaler = make_scalers(cfg)
    x_scaled, y_scaled = prepare_training_data(x_all, y_all, x_scaler, y_scaler)
    save_scalers(cfg, x_scaler, y_scaler, iteration)
    
    n_params = len(likelihood.varying_param_names)
    model = build_model_func(cfg, n_params=n_params)
    
    history, training_metrics = train_model_func(
        cfg=cfg,
        model=model,
        x_train=x_scaled,
        y_train=y_scaled,
        y_scaler=y_scaler,
        iteration=iteration,
        return_metrics=True
    )
    
    return x_scaler, y_scaler, training_metrics


def broadcast_scalers(x_scaler, y_scaler, using_mpi):
    if not using_mpi:
        return x_scaler, y_scaler
    
    comm = get_communicator()
    return comm.bcast(x_scaler, root=0), comm.bcast(y_scaler, root=0)


def run_sampling_step(cfg, likelihood, iteration, EmulatedLikelihood, run_sampler):
    if not is_master():
        return None, None, None, None
    
    model_path = os.path.join(cfg.trained_models_dir, f'trained_model_it_{iteration}.keras')
    x_scaler_path = os.path.join(cfg.scaler_dir, f'x_scaler_it_{iteration}.pkl')
    y_scaler_path = os.path.join(cfg.scaler_dir, f'y_scaler_it_{iteration}.pkl')
    chain_path = os.path.join(cfg.chains_dir, f'chain_it_{iteration}.h5') if cfg.save_chains else None
    
    emulated_likelihood = EmulatedLikelihood(model_path, x_scaler_path, y_scaler_path, true_likelihood=likelihood)
    samples, tempered_loglkls, sampling_metrics, sampler = run_sampler(cfg, emulated_likelihood, chain_path=chain_path, return_metrics=True, return_sampler=True)
    loglkls = tempered_loglkls * cfg.temperature
    
    return samples, loglkls, sampling_metrics, sampler


def run_resampling_step(cfg, likelihood, samples, loglkls, x_all, y_all, x_scaler, using_mpi, iteration, generate_resamples):
    print_master(f"Starting resampling for iteration {iteration}...")
    resampling_start = time.time()
    
    x_new, y_new, resampling_metrics = generate_resamples(
        cfg=cfg,
        samples=samples,
        loglkls=loglkls,
        true_likelihood=likelihood,
        x_all=x_all,
        y_all=y_all,
        x_scaler=x_scaler,
        return_metrics=True,
        use_mpi=using_mpi
    )
    
    if is_master():
        elapsed = time.time() - resampling_start
        n_accepted = len(x_new)
        print_master(f"Resampling completed in {elapsed:.2f}s")
        if n_accepted > 0:
            print_master(f"   Accepted {n_accepted} new samples ({elapsed/n_accepted:.2f}s per accepted sample)\n")
        else:
            print_master("")
    
    return x_new, y_new, resampling_metrics


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
    
    from model.model import build_model
    from training.training import train_model, save_training_data, load_training_data
    from sampling.sampler import run_sampler
    from sampling.resampler import generate_resamples
    from likelihood.surrogate import EmulatedLikelihood
    
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
    
    final_iteration = start_it
    final_converged = False
    
    for i in range(max_iterations):
        iteration_start = time.time()
        iteration = start_it + i
        final_iteration = iteration
        
        if use_convergence:
            print_master(f"--- Iteration {iteration} (max: {start_it + max_iterations - 1}) ---")
        else:
            print_master(f"--- Iteration {iteration}/{start_it + max_iterations - 1} ---")
        
        x_scaler, y_scaler = None, None
        if should_train_model(i, cfg):
            x_scaler, y_scaler, training_metrics = train_iteration_model(cfg, likelihood, x_all, y_all, iteration, build_model, train_model)
            
            if is_master():
                metrics_tracker.add_training_metrics(iteration=iteration, **training_metrics)
        else:
            if is_master():
                x_scaler, y_scaler = make_scalers(cfg)
                x_scaler.fit(x_all)
                y_scaler.fit(y_all.reshape(-1, 1))
                save_scalers(cfg, x_scaler, y_scaler, iteration)
        
        barrier()
        x_scaler, y_scaler = broadcast_scalers(x_scaler, y_scaler, using_mpi)
        barrier()
        
        samples, loglkls, sampling_metrics, sampler = run_sampling_step(cfg, likelihood, iteration, EmulatedLikelihood, run_sampler)
        
        if is_master():
            metrics_tracker.add_sampling_metrics(iteration=iteration, **sampling_metrics)
        
        converged = False
        if is_master():
            current_chain = sampler.get_chain()
            
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
                cfg, likelihood, samples, loglkls, x_all, y_all, x_scaler, using_mpi, iteration, generate_resamples
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