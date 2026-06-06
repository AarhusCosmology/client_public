# client.py

import time
import argparse
from pathlib import Path

import numpy as np

from config.config_loader import load_config_cli
from config.run_manager import (
    write_run_log,
    append_convergence_info,
    save_chain_summary,
    load_chain_summary,
)
from likelihood.base import build_likelihood
from likelihood.surrogate import SurrogateLikelihood
from metrics.convergence import build_convergence_metric
from metrics.metrics_tracker import MetricsTracker
from model.network import build_model, load_model
from sampling.initial_sampler import generate_samples
from sampling.sampler import build_sampler
from training.dataset import TrainingDataset
from training.losses import build_loss
from training.training import train_model, save_history
from utils.mpi_utils import (
    is_mpi_available,
    is_master,
    get_size,
    print_master,
    barrier,
    get_communicator,
    broadcast_and_evaluate,
)


def main():
    using_mpi = is_mpi_available()
    mpi_status = f"enabled ({get_size()} processes)" if using_mpi else "disabled (serial)"
    print_master(f"\nMPI: {mpi_status}\n")

    # ---- Arguments ----
    parser = argparse.ArgumentParser(
        description='CLiENT: Cosmological Likelihood Emulator using Neural networks with TensorFlow'
    )
    parser.add_argument('input_or_dir', help='Input YAML file (new run) or run directory (continue)')
    parser.add_argument('-n', '--name', help='Run name/tag for organisation (new runs only)')
    parser.add_argument('-o', '--output', default='results', help='Base output directory (new runs only)')
    parser.add_argument('-r', '--retrain', action='store_true', help='Force retrain even if a saved model exists')
    parser.add_argument('-s', '--start', type=int, help='Starting iteration (continue only, auto-detected if omitted)')
    parser.add_argument('-i', '--iterations', type=int, help='Number of (additional) iterations to run (overrides convergence criterion)')
    args = parser.parse_args()

    path = Path(args.input_or_dir)
    if path.is_dir() and (path / 'training_data').exists():
        args.mode = 'continue'
        args.run_dir = args.input_or_dir
    else:
        args.mode = 'default'
        args.input = args.input_or_dir

    # ---- Configuration (master loads cfg + metrics tracker, then bcasts cfg) ----
    if is_master():
        cfg = load_config_cli(args)
        metrics_tracker = MetricsTracker(cfg.run_dir, start_iteration=cfg.start_it)
        write_run_log(cfg, args.input if args.mode == 'default' else args.run_dir,
                      append=(cfg.run_mode != 'default'))
        print_master(f"Run:     {cfg.run_id}  [{cfg.run_mode}]")
        print_master(f"Results: {cfg.run_dir}\n")
    else:
        cfg = metrics_tracker = None

    if using_mpi:
        cfg = get_communicator().bcast(cfg, root=0)

    # ---- Likelihood ----
    print_master("Initialising likelihood...")
    likelihood = build_likelihood(cfg.wrapper, cfg.param)
    if cfg.n_sigma is not None:
        likelihood.restrict_prior_bounds(n_sigma=cfg.n_sigma)
    param_names  = likelihood.get_param_names()
    prior_bounds = likelihood.get_prior_bounds()
    ndim         = len(param_names)
    loglkl_fn    = lambda x: likelihood.loglkl({name: float(x[j]) for j, name in enumerate(param_names)})
    prior_lower  = [b[0] for b in prior_bounds.values()]
    prior_upper  = [b[1] for b in prior_bounds.values()]
    print_master(f"Parameters ({ndim}): {', '.join(param_names)}\n")

    # ---- Training data ----
    if cfg.run_mode != 'default':
        start_it = cfg.start_it
        if is_master():
            dataset = TrainingDataset.load(
                training_data_dir=cfg.training_data_dir,
                likelihood=likelihood,
                n_neighbors=cfg.n_neighbors,
                target_temperature=cfg.target_temperature,
                iteration=cfg.start_it,
            )
            print_master(f"Loaded {len(dataset.inputs)} training points (iteration {cfg.start_it}).\n")
        else:
            dataset = None
    else:
        start_it = 0
        print_master(f"Generating and evaluating {cfg.n_samples} initial samples via Latin hypercube...")
        t0 = time.time()
        x_init, y_init = broadcast_and_evaluate(
            lambda: generate_samples(likelihood=likelihood, n_samples=cfg.n_samples, strategy='lhs'),
            loglkl_fn,
            description="initial samples",
        )
        if is_master():
            valid = y_init > -1e20
            if (~valid).any():
                print_master(f"  Warning: filtered {(~valid).sum()}/{len(y_init)} outliers with log-lkl < -1e20")
                x_init, y_init = x_init[valid], y_init[valid]
            print_master(f"  Done in {time.time() - t0:.1f}s\n")
            dataset = TrainingDataset(
                inputs=x_init.astype(np.float32),
                targets=y_init.reshape(-1, 1).astype(np.float32),
                likelihood=likelihood,
                n_neighbors=cfg.n_neighbors,
                target_temperature=cfg.target_temperature,
            )
            dataset.save(cfg.training_data_dir / 'training_data_it_0.csv')
        else:
            dataset = None

    # ---- Iteration limits ----
    if cfg.n_it is not None:
        # If continuing, add one extra so the first (skip-retrain) iteration is not counted.
        extra           = 1 if cfg.run_mode != 'default' else 0
        n_iterations    = cfg.n_it + extra
        use_convergence = False
    else:
        n_iterations    = cfg.max_iterations
        use_convergence = cfg.convergence_enabled

    surrogate = sampler = None
    metric = build_convergence_metric(cfg.convergence_metric)
    prev_chain_summary = None
    if is_master() and cfg.run_mode != 'default' and cfg.start_it > 0:
        loaded = load_chain_summary(cfg.convergence_stats_dir, cfg.start_it - 1)
        if loaded is not None:
            prev_chain_summary, prev_path = loaded
            print_master(f"Loaded previous chain summary from {prev_path}")
    final_iteration = start_it
    final_converged = False

    # ===== Main loop =====
    for iteration in range(start_it, start_it + n_iterations):
        final_iteration = iteration
        t_iter          = time.time()

        if use_convergence:
            print_master(f"\n--- Iteration {iteration} (max {start_it + n_iterations - 1}) ---")
        else:
            print_master(f"\n--- Iteration {iteration}/{start_it + n_iterations - 1} ---")

        # -- Training --
        model_path = cfg.trained_models_dir / f'trained_model_it_{iteration}.keras'
        skip_train = (cfg.run_mode == 'skip_retrain_continue' and iteration == start_it)

        if not skip_train and is_master():
            if model_path.exists() and not cfg.retrain:
                print_master(f"Loading existing model from {model_path}")
            else:
                shuffle_idx = np.random.permutation(len(dataset.inputs))
                inputs  = dataset.inputs[shuffle_idx]
                targets = dataset.targets[shuffle_idx]

                model = build_model(
                    x_train=inputs,
                    n_layers=cfg.n_layers,
                    n_neurons=cfg.n_neurons,
                    activation=cfg.act_func,
                )
                loss = build_loss(
                    name=cfg.loss_func,
                    kappa=cfg.kappa_sigma,
                    n=ndim,
                    y_global_max=float(targets.max()),
                )
                history, training_metrics = train_model(
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
                model.save(model_path)
                save_history(history.history,
                             cfg.training_history_dir / f'history_it_{iteration}.csv')
                print_master(f"  Model saved to {model_path}")
                metrics_tracker.add_training_metrics(iteration=iteration, **training_metrics)

        barrier()

        # -- Sampling --
        chain = logposts = None
        if is_master():
            model = load_model(model_path)

            if surrogate is None:
                surrogate = SurrogateLikelihood(likelihood, model)
                sampler   = build_sampler(
                    name=cfg.sampler,
                    n_walkers=cfg.n_walkers,
                    ndim=ndim,
                    logpost_fn=lambda positions: surrogate.logpost(positions) / cfg.temperature,
                    bounds=(prior_lower, prior_upper),
                )
            else:
                # Update weights in-place to preserve the compiled XLA graph.
                surrogate.model.set_weights(model.get_weights())

            print_master(f"Sampling (method={cfg.sampler}, {cfg.n_walkers} chains, T={cfg.temperature})...")
            t_sample = time.time()

            initial_pos = np.random.uniform(
                low =prior_lower,
                high=prior_upper,
                size=(cfg.n_walkers, ndim),
            )
            sampler.run(
                initial_pos=initial_pos,
                max_steps=cfg.max_steps,
            )

            chain      = sampler.get_chain(discard=cfg.burn_in, flat=True).copy()
            logposts   = sampler.get_logpost(discard=cfg.burn_in, flat=True).copy()
            acceptance = sampler.get_acceptance_fraction()
            # Free the raw chain immediately — it's 2+ GB and no longer needed.
            sampler.free_memory()

            sampling_time = time.time() - t_sample
            steps_done    = sampler.get_n_steps()

            print_master(f"  {len(chain)} samples in {sampling_time:.1f}s ({steps_done} steps/walker)")
            metrics_tracker.add_sampling_metrics(
                iteration=iteration,
                steps_per_walker=int(steps_done),
                acceptance_rate=acceptance,
                sampling_time=sampling_time,
            )

        # -- Convergence check --
        converged = False
        if is_master():
            chain_summary = metric.summarise(chain)
            save_chain_summary(cfg.convergence_stats_dir, iteration, chain_summary)

            if prev_chain_summary is not None:
                r_minus_one = metric.compute_from_summary(chain, prev_chain_summary)
                converged = r_minus_one < cfg.convergence_threshold
                print_master(f"  R-1 = {r_minus_one:.6f}  (threshold: {cfg.convergence_threshold})")
                metrics_tracker.add_convergence_metrics(iteration, r_minus_one, converged)
                if use_convergence and converged:
                    print_master(f"\nConverged at iteration {iteration}!\n")
                    final_converged = True
            else:
                print_master("  R-1 not yet calculable (need >= 2 iterations)")
            prev_chain_summary = chain_summary

        if use_convergence:
            if using_mpi:
                converged = get_communicator().bcast(converged, root=0)
            if converged:
                if is_master():
                    metrics_tracker.add_iteration_metrics(iteration, time.time() - t_iter)
                    metrics_tracker.save_all_metrics()
                break

        # -- Resampling (skip on the last iteration) --
        if iteration < start_it + n_iterations - 1:
            t_resamp = time.time()
            n_before = len(dataset.inputs) if is_master() else None

            selected, log_L_selected = broadcast_and_evaluate(
                lambda: dataset.select_candidates(
                    chain=chain,
                    logposts=logposts.astype(np.float64),
                    surrogate=surrogate,
                    n_augment=cfg.n_augment,
                    sampling_temperature=cfg.temperature,
                    pool_factor=cfg.pool_factor,
                ),
                loglkl_fn,
                description="augmentation candidates",
            )

            if is_master():
                dataset.add_evaluated_points(selected, log_L_selected)

                n_added     = len(dataset.inputs) - n_before
                resamp_time = time.time() - t_resamp

                print_master(f"  Resampling: +{n_added} points in {resamp_time:.1f}s")
                dataset.save(cfg.training_data_dir / f'training_data_it_{iteration + 1}.csv')
                metrics_tracker.add_resampling_metrics(
                    iteration=iteration,
                    pool_size=min(cfg.pool_factor * cfg.n_augment, len(chain)),
                    n_evaluated=len(selected),
                    n_added=n_added,
                    resampling_time=resamp_time,
                )

        if is_master():
            metrics_tracker.add_iteration_metrics(iteration, time.time() - t_iter)
            metrics_tracker.save_progress_metrics(iteration)

    # ---- Finalise ----
    barrier()

    if is_master():
        metrics_tracker.save_all_metrics()
        append_convergence_info(cfg.run_dir, final_iteration, final_converged)
        print_master(f"\nRun complete: {cfg.run_id}")
        print_master(f"Results:      {cfg.run_dir}")


if __name__ == "__main__":
    main()
