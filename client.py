import time
import argparse
import numpy as np

from likelihood.base import build_likelihood
from utils.mpi_utils import (
    is_mpi_available,
    is_master,
    get_size,
    print_master,
    bcast,
    broadcast_and_evaluate
)

def main():
    using_mpi = is_mpi_available()
    master = is_master()
    mpi_status = f"Enabled ({get_size()} processes)" if using_mpi else "Disabled"
    print_master(f"\nMPI status: {mpi_status}\n")

    # ---- Arguments ---- 
    parser = argparse.ArgumentParser()
    parser.add_argument('input_or_dir', help='Input YAML file (new run) or run directory (continue)')
    parser.add_argument('-n', '--name', help='Run name/tag for organisation (new runs only)')
    parser.add_argument('-o', '--output', default='results', help='Base output directory (new runs only)')
    parser.add_argument('-r', '--retrain', action='store_true', help='Force retrain even if a saved model exists')
    parser.add_argument('-s', '--start', type=int, help='Starting iteration (continue only, auto-detected if omitted)')
    parser.add_argument('-i', '--iterations', type=int, help='Number of (additional) iterations to run (overrides convergence criterion)')
    args = parser.parse_args()

    # ---- Configuration (master loads the run and broadcasts it to workers) ----
    if master:
        from run.run import Run
        from run.metrics import MetricsTracker

        run = Run.from_args(args)
        if run.is_new:
            run.create_directories(args.input_or_dir)
        
        metrics_tracker = MetricsTracker(
            results_dir=run.run_dir,
            start_iteration=run.start_iteration,
            preserve_start_metrics=False if run.is_new else True
        )

        print_master(f"Run ID: {run.run_id}")
        print_master(f"Run mode: {run.mode}")
        print_master(f"Run directory: {run.run_dir}\n")
    else:
        run = None
        metrics_tracker = None

    run = bcast(run)
    cfg = run.config

    # ---- Likelihood ----
    print_master("Initialising likelihood...")

    likelihood = build_likelihood(
        wrapper = cfg.likelihood.wrapper,
        input_path = cfg.likelihood.input,
    )
    if cfg.prior.n_sigma is not None:
        likelihood.restrict_prior_bounds(cfg.prior.n_sigma)
    
    # ---- Surrogate metadata ----
    if master:
        from likelihood.surrogate import SurrogateMetadata, SurrogateLikelihood
        from dataset.dataset import TrainingDataset
        surrogate_metadata = SurrogateMetadata.from_likelihood(likelihood)
        if run.is_new:
            surrogate_metadata.save(run.run_dir / 'metadata.json')
    else:
        surrogate_metadata = None

    # ---- Training data (new run) ----
    if run.is_new:
        print_master(f"Generating and evaluating {cfg.prior.n_samples} {cfg.prior.sampling_strategy} samples")

        if master:
            from sampling.prior_sampler import sample_prior
            start_time = time.time()
            prior_samples = sample_prior(
                likelihood=likelihood,
                n_samples=cfg.prior.n_samples,
                strategy=cfg.prior.sampling_strategy
            )
        else:
            prior_samples = None
        
        inputs, targets = broadcast_and_evaluate(
            samples=prior_samples,
            evaluator=likelihood.loglkl
        )

        if master:
            valid = np.isfinite(targets)
            if not valid.all():
                inputs, targets = inputs[valid], targets[valid]
                print_master(f"Warning: filtered {valid.size - np.count_nonzero(valid)} targets with non-finite loglkl values")
            print_master(f"Finished in {time.time() - start_time:.1f}s\n")

            dataset = TrainingDataset(
                inputs=inputs,
                targets=targets,
                likelihood=likelihood,
                n_neighbors=cfg.acquisition.n_neighbors,
                target_temperature=cfg.acquisition.target_temperature
            )
            dataset.save(run.training_data_dir / 'data_it_0.csv')
        else:
            dataset = None

    # --- Training data (continue run) ----
    else:
        if master:
            dataset = TrainingDataset.load(
                training_data_dir=run.training_data_dir,
                likelihood=likelihood,
                n_neighbors=cfg.acquisition.n_neighbors,
                target_temperature=cfg.acquisition.target_temperature,
                iteration=run.start_iteration
            )
            print_master(f"Loaded {len(dataset.inputs)} training samples from data_it_{run.start_iteration}.csv")
        else:
            dataset = None

    # ---- Convergence metric ----
    previous_chain_summary = None
    if master:
        from convergence.convergence import build_convergence_metric
        metric = build_convergence_metric(cfg.convergence.metric)
        if not run.is_new and run.start_iteration > 0:
            previous_chain_summary = metric.load_chain_summary(run.convergence_stats_dir, run.start_iteration - 1)
            print_master(f"Loaded previous chain summary from chain_summary_it_{run.start_iteration - 1}.npz")
    else:
        metric = None

    final_iteration = run.final_iteration
    use_convergence = run.use_convergence

    # Hoist loop-invariant values and master-only imports out of the iteration loop.
    if master:
        ndim = likelihood.ndim
        prior_bounds = np.asarray(list(likelihood.prior_bounds.values()), dtype=float)
        prior_lower = prior_bounds[:, 0]
        prior_upper = prior_bounds[:, 1]
        inverse_sampling_temperature = 1.0 / cfg.sampling.temperature

        import tensorflow as tf
        from dataset.acquisition import select_points
        from model.network import build_model, load_model
        from sampling.sampler import build_sampler
        from training.losses import build_loss
        from training.training import train_model, save_history

    # ---- Main loop ----
    for iteration in range(run.start_iteration, final_iteration + 1):
        print_master(f"\n--- Iteration {iteration}/{final_iteration} ---")
        if master:
            iteration_start_time = time.time()
            model_path = run.trained_models_dir / f'model_it_{iteration}.keras'
            # Train for new/retrain runs, after acquisition updates, or if a continued run has no saved model.
            should_train = (
                run.is_new
                or run.retrain
                or iteration != run.start_iteration
                or not model_path.exists()
            )
            if should_train:
                if not model_path.exists() and not run.is_new:
                    print_master(f"No saved model found for iteration {iteration}; retraining...")
                print_master(f"Training model on {len(dataset.inputs)} samples...")
            else:
                print_master(f"Loading existing model: {model_path}")

        # ---- Training ----
        if master and should_train:
            tf.keras.backend.clear_session()

            # Shuffle the training data to avoid biasing the validation set
            shuffle_indices = np.random.permutation(len(dataset.inputs))
            inputs, targets = dataset.inputs[shuffle_indices], dataset.targets[shuffle_indices]

            model = build_model(
                inputs=inputs,
                targets=targets,
                n_layers=cfg.model.n_layers,
                n_neurons=cfg.model.n_neurons,
                activation=cfg.model.activation
            )
            loss = build_loss(
                name=cfg.training.loss,
                sigma_level=cfg.training.sigma_level,
                chi2_dof=ndim,
                max_loglkl=float(targets.max())
            )
            history, training_metrics = train_model(
                model=model,
                inputs=inputs,
                targets=targets,
                loss=loss,
                learning_rate=cfg.training.learning_rate,
                n_epochs=cfg.training.n_epochs,
                batch_size=cfg.training.batch_size,
                validation_split=cfg.training.validation_split,
                patience=cfg.training.patience
            )
            model.save(model_path)
            save_history(
                history.history,
                run.training_history_dir / f'history_it_{iteration}.csv'
            )
            metrics_tracker.add_training_metrics(
                iteration=iteration,
                epoch=training_metrics['epoch'],
                train_loss=training_metrics['train_loss'],
                val_loss=training_metrics['val_loss'],
                training_time=training_metrics['training_time']
            )

        # ---- Surrogate ----
        if master:
            # Reuse the freshly trained model in memory; only load from disk for already-trained iterations.
            if not should_train:
                tf.keras.backend.clear_session()
                model = load_model(model_path)
            surrogate = SurrogateLikelihood(
                model=model,
                metadata=surrogate_metadata,
            )

            # ---- Sampling ----
            def tempered_logpost_fn(positions):
                return surrogate.logpost(positions) * inverse_sampling_temperature

            sampler = build_sampler(
                name=cfg.sampling.sampler,
                n_walkers=cfg.sampling.n_walkers,
                ndim=ndim,
                log_prob_fn=tempered_logpost_fn
            )
            initial_positions = np.random.uniform(
                low=prior_lower,
                high=prior_upper,
                size=(cfg.sampling.n_walkers, ndim)
            )
            sampling_start_time = time.time()
            sampler.run(
                n_steps=cfg.sampling.n_steps,
                initial_positions=initial_positions,
            )
            sampling_elapsed_time = time.time() - sampling_start_time

            chain = sampler.chain(discard=cfg.sampling.burn_in).numpy()
            logposts = sampler.log_prob(discard=cfg.sampling.burn_in).numpy()
            acceptance = sampler.acceptance_fraction().numpy()
            sampler.reset()

            mean_acceptance = float(np.mean(acceptance))
            print_master(f"Finished sampling in {sampling_elapsed_time:.1f}s (acceptance rate: {mean_acceptance:.3f})")

            metrics_tracker.add_sampling_metrics(
                iteration=iteration,
                steps_per_walker=cfg.sampling.n_steps,
                acceptance_rate=mean_acceptance,
                sampling_time=sampling_elapsed_time
            )
        
        # ---- Convergence check ----
        converged = False
        if master:
            chain_summary = metric.summarise(chain)
            metric.save_chain_summary(
                convergence_stats_dir=run.convergence_stats_dir,
                iteration=iteration,
                chain_summary=chain_summary
            )
            if previous_chain_summary is None:
                print_master(f"No previous chain summary available, skipping convergence check")
            else:
                metric_value = metric.compute_from_summaries(
                    current_chain_summary=chain_summary,
                    previous_chain_summary=previous_chain_summary
                )
                print_master(f"{metric.name}: {metric_value:.4e} (threshold: {cfg.convergence.threshold:.4e})")
                converged = metric_value < cfg.convergence.threshold
                metrics_tracker.add_convergence_metrics(
                    iteration=iteration,
                    metric_value=metric_value,
                    converged=converged,
                    metric_name=metric.name
                )
                if use_convergence and converged:
                    print_master(f"Convergence criterion met, stopping...")
            previous_chain_summary = chain_summary

        # Broadcast the stopping decision so all ranks leave the loop together.
        if use_convergence:
            converged = bcast(converged)
            if converged:
                if master:
                    iteration_elapsed_time = time.time() - iteration_start_time
                    metrics_tracker.add_iteration_metrics(
                        iteration=iteration,
                        iteration_time=iteration_elapsed_time,
                    )
                    metrics_tracker.save_progress_metrics(iteration)
                break
        
        # ---- Acquisition ----
        if iteration < final_iteration:
            print_master(f"Selecting and evaluating {cfg.acquisition.n_append} new samples from surrogate chain")

            if master:
                acquisition_start_time = time.time()
                new_samples = select_points(
                    dataset=dataset,
                    chain=chain,
                    logposts=logposts,
                    surrogate=surrogate,
                    n_append=cfg.acquisition.n_append,
                    mcmc_temperature=cfg.sampling.temperature,
                    pool_factor=cfg.acquisition.pool_factor
                )
            else:
                new_samples = None

            new_inputs, new_targets = broadcast_and_evaluate(
                samples=new_samples,
                evaluator=likelihood.loglkl
            )

            if master:
                valid = np.isfinite(new_targets)
                if not valid.all():
                    new_inputs, new_targets = new_inputs[valid], new_targets[valid]
                    print_master(f"Warning: filtered {valid.size - np.count_nonzero(valid)} targets with non-finite loglkl values")

                n_current_inputs = len(dataset.inputs)
                dataset.add_data(
                    inputs=new_inputs,
                    targets=new_targets
                )
                dataset.save(run.training_data_dir / f'data_it_{iteration + 1}.csv')
                n_new_inputs = len(dataset.inputs) - n_current_inputs
                acquisition_elapsed_time = time.time() - acquisition_start_time
                print_master(f"Added {n_new_inputs} new training samples in {acquisition_elapsed_time:.1f}s for a total of {len(dataset.inputs)} samples")
                metrics_tracker.add_acquisition_metrics(
                    iteration=iteration,
                    n_evaluated=len(new_inputs),
                    n_added=n_new_inputs,
                    acquisition_time=acquisition_elapsed_time
                )
        if master:
            iteration_elapsed_time = time.time() - iteration_start_time
            metrics_tracker.add_iteration_metrics(
                iteration=iteration,
                iteration_time=iteration_elapsed_time,
            )
            metrics_tracker.save_progress_metrics(iteration)

    # ---- Finalisation ----
    if master:
        metrics_tracker.save_all_metrics()
        print_master(f"\nRun completed: {run.run_id}")
        print_master(f"Results saved in: {run.run_dir}")

if __name__ == "__main__":
    main()
