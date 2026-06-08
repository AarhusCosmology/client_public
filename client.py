import time
import argparse
import numpy as np

from config.run import Run
from likelihood.base import build_likelihood
from likelihood.surrogate import SurrogateLikelihood, SurrogateMetadata
from metrics.convergence import (
    build_convergence_metric,
    save_chain_summary,
    load_chain_summary,
)
from metrics.metrics_tracker import MetricsTracker
from model.network import build_model, load_model
from sampling.prior_sampler import sample_prior
from sampling.sampler import build_sampler
from training.dataset import TrainingDataset
from training.acquisition import select_candidates
from training.losses import build_loss
from training.training import train_model, save_history
from utils.mpi_utils import (
    is_mpi_available,
    is_master,
    get_size,
    print_master,
    barrier,
    bcast,
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

    # ---- Configuration (master loads the run, then bcasts it to workers) ----
    if is_master():
        run = Run.from_args(args)
        if not run.is_continuation:
            run.create_directories(args.input_or_dir)
        metrics_tracker = MetricsTracker(run.run_dir, start_iteration=run.start_iteration)
        print_master(f"Run:     {run.run_id}  [{run.mode_label}]")
        print_master(f"Results: {run.run_dir}\n")
    else:
        run = metrics_tracker = None

    run = bcast(run)

    config   = run.config
    start_it = run.start_iteration

    # ---- Likelihood ----
    print_master("Initialising likelihood...")
    likelihood = build_likelihood(config.likelihood.wrapper, config.likelihood.input)
    if config.data.n_sigma is not None:
        likelihood.restrict_prior_bounds(n_sigma=config.data.n_sigma)
    param_names  = likelihood.get_param_names()
    prior_bounds = likelihood.get_prior_bounds()
    ndim         = len(param_names)
    loglkl_fn    = likelihood.loglkl
    prior_lower  = [b[0] for b in prior_bounds.values()]
    prior_upper  = [b[1] for b in prior_bounds.values()]
    print_master(f"Parameters ({ndim}): {', '.join(param_names)}\n")

    surrogate_metadata = None
    if is_master():
        surrogate_metadata = SurrogateMetadata.from_likelihood(likelihood)
        if not run.is_continuation:
            surrogate_metadata.save(run.run_dir / 'metadata.json')

    # ---- Training data ----
    if run.is_continuation:
        if is_master():
            dataset = TrainingDataset.load(
                training_data_dir=run.training_data,
                likelihood=likelihood,
                n_neighbors=config.data.n_neighbors,
                target_temperature=config.data.target_temperature,
                iteration=start_it,
            )
            print_master(f"Loaded {len(dataset.inputs)} training points (iteration {start_it}).\n")
        else:
            dataset = None
    else:
        print_master(f"Generating and evaluating {config.data.n_samples} initial samples via Latin hypercube...")
        t0 = time.time()
        x_init, y_init = broadcast_and_evaluate(
            lambda: sample_prior(likelihood=likelihood, n_samples=config.data.n_samples, strategy='lhs'),
            loglkl_fn,
            description="initial samples",
        )
        if is_master():
            valid = np.isfinite(y_init)
            if (~valid).any():
                print_master(f"  Warning: filtered {(~valid).sum()}/{len(y_init)} samples with non-finite log-lkl")
                x_init, y_init = x_init[valid], y_init[valid]
            print_master(f"  Done in {time.time() - t0:.1f}s\n")
            dataset = TrainingDataset(
                inputs=x_init.astype(np.float32),
                targets=y_init.reshape(-1, 1).astype(np.float32),
                likelihood=likelihood,
                n_neighbors=config.data.n_neighbors,
                target_temperature=config.data.target_temperature,
            )
            dataset.save(run.training_data / 'training_data_it_0.csv')
        else:
            dataset = None

    # ---- Iteration limits ----
    n_iterations    = run.n_iterations
    use_convergence = run.use_convergence

    surrogate = sampler = None
    metric = build_convergence_metric(config.convergence.metric)
    prev_chain_summary = None
    if is_master() and run.is_continuation and start_it > 0:
        loaded = load_chain_summary(run.convergence_stats, start_it - 1)
        if loaded is not None:
            prev_chain_summary, prev_path = loaded
            print_master(f"Loaded previous chain summary from {prev_path}")

    # ===== Main loop =====
    for iteration in range(start_it, start_it + n_iterations):
        t_iter          = time.time()

        if use_convergence:
            print_master(f"\n--- Iteration {iteration} (max {run.last_iteration}) ---")
        else:
            print_master(f"\n--- Iteration {iteration}/{run.last_iteration} ---")

        # -- Training --
        model_path = run.trained_models / f'trained_model_it_{iteration}.keras'
        skip_train = (run.reuse_initial_model and iteration == start_it)

        if not skip_train and is_master():
            if model_path.exists() and not run.retrain:
                print_master(f"Loading existing model from {model_path}")
            else:
                shuffle_idx = np.random.permutation(len(dataset.inputs))
                inputs  = dataset.inputs[shuffle_idx]
                targets = dataset.targets[shuffle_idx]

                model = build_model(
                    x_train=inputs,
                    n_layers=config.model.n_layers,
                    n_neurons=config.model.n_neurons,
                    activation=config.model.activation,
                )
                loss = build_loss(
                    name=config.training.loss,
                    kappa=config.training.kappa_sigma,
                    n=ndim,
                    y_global_max=float(targets.max()),
                )
                history, training_metrics = train_model(
                    model=model,
                    inputs=inputs,
                    targets=targets,
                    loss=loss,
                    learning_rate=config.training.learning_rate,
                    n_epochs=config.training.n_epochs,
                    batch_size=config.training.batch_size,
                    validation_split=config.training.validation_split,
                    patience=config.training.patience,
                    return_metrics=True,
                )
                model.save(model_path)
                save_history(history.history,
                             run.training_history / f'history_it_{iteration}.csv')
                print_master(f"  Model saved to {model_path}")
                metrics_tracker.add_training_metrics(iteration=iteration, **training_metrics)

        barrier()

        # -- Sampling --
        chain = logposts = None
        if is_master():
            model = load_model(model_path)

            if surrogate is None:
                surrogate = SurrogateLikelihood(model, surrogate_metadata)
                sampler   = build_sampler(
                    name=config.sampling.sampler,
                    n_walkers=config.sampling.n_walkers,
                    n_chains=config.sampling.n_chains,
                    ndim=ndim,
                    logpost_fn=lambda positions: surrogate.logpost(positions) / config.sampling.temperature,
                    bounds=(prior_lower, prior_upper),
                )
            else:
                # Update weights in-place to preserve the compiled XLA graph.
                surrogate.model.set_weights(model.get_weights())

            print_master(f"Sampling (method={config.sampling.sampler}, {config.sampling.n_chains} chains × {config.sampling.n_walkers} walkers, T={config.sampling.temperature})...")
            t_sample = time.time()

            initial_pos = np.random.uniform(
                low =prior_lower,
                high=prior_upper,
                size=(config.sampling.n_chains, config.sampling.n_walkers, ndim),
            )
            sampler.run(
                initial_pos=initial_pos,
                max_steps=config.sampling.max_steps,
            )

            chain      = sampler.get_chain(discard=config.sampling.burn_in, flat=True).copy()
            logposts   = sampler.get_logpost(discard=config.sampling.burn_in, flat=True).copy()
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
            save_chain_summary(run.convergence_stats, iteration, chain_summary)

            if prev_chain_summary is not None:
                metric_value = metric.compute_from_summary(chain, prev_chain_summary)
                converged = metric_value < config.convergence.threshold
                print_master(f"  {metric.name} = {metric_value:.6f}  (threshold: {config.convergence.threshold})")
                metrics_tracker.add_convergence_metrics(iteration, metric_value, converged, metric_name=metric.name)
                if use_convergence and converged:
                    print_master(f"\nConverged at iteration {iteration}!\n")
            else:
                print_master("  R-1 not yet calculable (need >= 2 iterations)")
            prev_chain_summary = chain_summary

        if use_convergence:
            converged = bcast(converged)
            if converged:
                if is_master():
                    metrics_tracker.add_iteration_metrics(iteration, time.time() - t_iter)
                    metrics_tracker.save_all_metrics()
                break

        # -- Resampling (skip on the last iteration) --
        if iteration < run.last_iteration:
            t_resamp = time.time()
            n_before = len(dataset.inputs) if is_master() else None

            selected, log_L_selected = broadcast_and_evaluate(
                lambda: select_candidates(
                    dataset,
                    chain=chain,
                    logposts=logposts.astype(np.float64),
                    surrogate=surrogate,
                    n_augment=config.data.n_augment,
                    sampling_temperature=config.sampling.temperature,
                    pool_factor=config.data.pool_factor,
                ),
                loglkl_fn,
                description="augmentation candidates",
            )

            if is_master():
                dataset.add_evaluated_points(selected, log_L_selected)

                n_added     = len(dataset.inputs) - n_before
                resamp_time = time.time() - t_resamp

                print_master(f"  Resampling: +{n_added} points in {resamp_time:.1f}s")
                dataset.save(run.training_data / f'training_data_it_{iteration + 1}.csv')
                metrics_tracker.add_resampling_metrics(
                    iteration=iteration,
                    pool_size=min(config.data.pool_factor * config.data.n_augment, len(chain)),
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
        print_master(f"\nRun complete: {run.run_id}")
        print_master(f"Results:      {run.run_dir}")


if __name__ == "__main__":
    main()
