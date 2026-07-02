# benchmarking/benchmark.py

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark surrogate likelihood")
    parser.add_argument("run_dir", help="Path to run directory")
    parser.add_argument(
        "-i",
        "--iteration",
        type=int,
        default=None,
        help="Iteration to benchmark (auto-detects latest if not specified)",
    )
    parser.add_argument(
        "-n",
        "--n-steps",
        type=int,
        default=None,
        help="Number of MCMC steps (defaults to max_steps from config)",
    )
    parser.add_argument("-t", "--thin", type=int, default=1, help="Thinning factor for chains")
    parser.add_argument("-p", "--params", nargs="+", default=None, help="Parameter indices to include in analysis")
    parser.add_argument(
        "-c",
        "--chains",
        default=None,
        help="Path to MontePython or Cobaya chains directory for comparison",
    )
    parser.add_argument("--no-training-data", action="store_true", help="Skip loading training data visualization")
    parser.add_argument("--no-training-history", action="store_true", help="Skip loading training history")
    return parser.parse_args()


def load_run_config(run_dir):
    import yaml

    yaml_files = list(run_dir.glob("*.yaml"))
    if not yaml_files:
        raise FileNotFoundError(f"No .yaml file found in {run_dir}")
    if len(yaml_files) > 1:
        print(f"Warning: Multiple .yaml files found in {run_dir}. Using {yaml_files[0].name}")

    config_yaml = yaml_files[0]
    with open(config_yaml, encoding="utf-8") as f:
        return config_yaml, yaml.safe_load(f)


def resolve_iteration(run_dir, requested_iteration):
    if requested_iteration is not None:
        return requested_iteration

    trained_models_dir = run_dir / "trained_models"
    if not trained_models_dir.exists():
        raise FileNotFoundError(f"trained_models directory not found in {run_dir}")

    model_files = list(trained_models_dir.glob("trained_model_it_*.keras"))
    if not model_files:
        raise FileNotFoundError(f"No trained models found in {trained_models_dir}")

    iterations = []
    for model_file in model_files:
        try:
            iterations.append(int(model_file.stem.split("_")[-1]))
        except ValueError:
            continue

    if not iterations:
        raise ValueError(f"Could not parse iteration numbers from model files in {trained_models_dir}")

    iteration = max(iterations)
    print(f"Auto-detected latest iteration: {iteration}")
    return iteration


def load_surrogate(run_dir, iteration):
    from likelihood.surrogate import SurrogateLikelihood, SurrogateMetadata
    from model.network import load_model

    metadata = SurrogateMetadata.load(run_dir / "metadata.json")
    model = load_model(run_dir / f"trained_models/trained_model_it_{iteration}.keras")
    return SurrogateLikelihood(model, metadata), metadata


def load_training_samples(run_dir, iteration, param_names, skip_training_data):
    import pandas as pd

    if skip_training_data:
        return None

    data_path = run_dir / "training_data" / f"training_data_it_{iteration}.csv"
    if not data_path.exists():
        return None

    return pd.read_csv(data_path)[param_names].to_numpy()


def load_or_run_chain(run_dir, iteration, sampling_config, n_steps, thin, prior_bounds, param_names, surrogate):
    import numpy as np

    from sampling.sampler import build_sampler

    sampler_name = sampling_config["sampler"]
    output_dir = run_dir / "benchmark_chains"
    output_dir.mkdir(exist_ok=True)
    chain_path = output_dir / (
        f"benchmark_chain_it_{iteration}_{sampler_name}_"
        f"steps{n_steps}_thin{thin}.npz"
    )

    if chain_path.exists():
        data = np.load(chain_path)
        chain = data["chain"]
        log_prob = data["log_prob"]
        print(f"Loaded chain: {chain.shape}")
        return chain_path, chain, log_prob

    n_walkers = sampling_config["n_walkers"]
    n_chains = sampling_config["n_chains"]
    burn_in = sampling_config["burn_in"]
    print(f"Running {sampler_name} sampler: {n_walkers} walkers, {burn_in} burn-in, {n_steps} steps")

    lower = [bounds[0] for bounds in prior_bounds.values()]
    upper = [bounds[1] for bounds in prior_bounds.values()]
    sampler = build_sampler(
        name=sampler_name,
        n_walkers=n_walkers,
        n_chains=n_chains,
        ndim=len(param_names),
        logpost_fn=surrogate.logpost,
        bounds=(lower, upper),
    )
    initial_pos = np.random.uniform(
        low=lower,
        high=upper,
        size=(n_chains, n_walkers, len(param_names)),
    )
    sampler.run(initial_pos=initial_pos, max_steps=n_steps)
    chain = sampler.get_chain(discard=burn_in, thin=thin)
    log_prob = sampler.get_logpost(discard=burn_in, thin=thin)
    np.savez(chain_path, chain=chain, log_prob=log_prob)
    print(f"Chain shape: {chain.shape}")
    return chain_path, chain, log_prob


def make_getdist_samples(chain, log_prob, sampler_name, getdist_names, param_labels, getdist_ranges):
    from getdist import MCSamples

    from benchmarking.getdist_utils import getdist_sample_inputs

    samples, loglikes, convergence_available = getdist_sample_inputs(chain, log_prob, sampler_name)
    return (
        MCSamples(
            samples=samples,
            names=getdist_names,
            labels=param_labels,
            loglikes=loglikes,
            ranges=getdist_ranges,
        ),
        convergence_available,
    )


def load_reference_samples(chains_path, param_names, getdist_names, param_labels, getdist_ranges, scales):
    from getdist import MCSamples

    from benchmarking.chains import detect_chain_format, load_cobaya_chains, load_montepython_chains

    if not chains_path:
        return None, None

    try:
        chain_format = detect_chain_format(chains_path)
        print(f"Detected {chain_format} chain format")

        if chain_format == "cobaya":
            samples_list, loglikes_list = load_cobaya_chains(chains_path, param_names, thin=1)
        else:
            samples_list, loglikes_list = load_montepython_chains(
                chains_path,
                param_names,
                thin=1,
                scales=scales,
            )

        reference_samples = MCSamples(
            samples=samples_list,
            names=getdist_names,
            labels=param_labels,
            loglikes=[-loglikes for loglikes in loglikes_list],
            ranges=getdist_ranges,
        )
        print(f"Loaded {sum(len(samples) for samples in samples_list)} samples from {chain_format} chains")
        return reference_samples, chain_format
    except Exception as e:
        print(f"Warning: Could not load chains: {e}")
        return None, None


def main():
    args = parse_args()

    from benchmarking.diagnostics import DiagnosticsConfig, TeeOutput, print_diagnostics
    from benchmarking.getdist_utils import (
        getdist_names_for_params,
        getdist_ranges_for_params,
        select_plot_params,
    )
    from benchmarking.plotting import save_training_history_plot, save_triangle_plot

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_dir)

    config_yaml, config = load_run_config(run_dir)
    n_steps = config["sampling"]["max_steps"] if args.n_steps is None else args.n_steps
    iteration = resolve_iteration(run_dir, args.iteration)
    surrogate, metadata = load_surrogate(run_dir, iteration)

    param_names = surrogate.get_param_names()
    prior_bounds = surrogate.get_prior_bounds()
    param_labels = surrogate.get_param_labels()
    getdist_names = getdist_names_for_params(param_names)
    getdist_ranges = getdist_ranges_for_params(param_names, getdist_names, prior_bounds)

    training_samples = load_training_samples(run_dir, iteration, param_names, args.no_training_data)
    sampler_name = config["sampling"]["sampler"]
    chain_path, chain, log_prob = load_or_run_chain(
        run_dir,
        iteration,
        config["sampling"],
        n_steps,
        args.thin,
        prior_bounds,
        param_names,
        surrogate,
    )
    samples, surrogate_convergence_available = make_getdist_samples(
        chain,
        log_prob,
        sampler_name,
        getdist_names,
        param_labels,
        getdist_ranges,
    )

    plot_params, param_indices = select_plot_params(args.params, param_names, getdist_names)
    reference_samples, reference_sampler = load_reference_samples(
        args.chains,
        param_names,
        getdist_names,
        param_labels,
        getdist_ranges,
        metadata.scales,
    )

    diagnostics_config = DiagnosticsConfig(
        iteration=iteration,
        config_yaml=config_yaml,
        run_dir=run_dir,
        thin=args.thin,
        n_steps=n_steps,
        chains_path=args.chains,
        surrogate_sampler=sampler_name,
        reference_sampler=reference_sampler,
        surrogate_convergence_available=surrogate_convergence_available,
    )

    log_file_path = run_dir / "benchmark_results" / f"{timestamp}_diagnostics_it_{iteration}.log"
    with TeeOutput(str(log_file_path)):
        print_diagnostics(
            samples,
            reference_samples,
            param_names,
            getdist_names,
            diagnostics_config,
            surrogate,
        )

    figure_dir = run_dir / "benchmark_figures"
    save_triangle_plot(
        figure_dir,
        timestamp,
        iteration,
        samples,
        reference_samples,
        plot_params,
        param_indices,
        getdist_ranges,
        training_samples=training_samples,
        surrogate_sampler=sampler_name,
        reference_sampler=reference_sampler,
    )

    if not args.no_training_history:
        save_training_history_plot(
            figure_dir,
            timestamp,
            iteration,
            run_dir / f"training_history/history_it_{iteration}.csv",
        )

    print(f"\n=== Benchmark Complete ===\nChain: {chain_path}\nPlots: {figure_dir}\nLog: {log_file_path}")


if __name__ == "__main__":
    main()
