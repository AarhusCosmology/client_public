# benchmarking/benchmark.py

import os
import re
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
from datetime import datetime
from getdist import MCSamples, plots
from matplotlib.lines import Line2D
from scipy import stats
from likelihood.surrogate import SurrogateLikelihood, SurrogateMetadata
from model.network import load_model
from sampling.sampler import build_sampler

textwidth_pts = 440
width_inches = textwidth_pts / 72.27
fontsize = 11 / 1.2

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': 'cmr10',
    'font.size': fontsize,
    'mathtext.fontset': 'cm',
    'axes.formatter.use_mathtext': True,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.6,
    'ytick.minor.width': 0.6,
    'lines.linewidth': 1.5,
    'patch.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'savefig.dpi': 300,
})


class TeeOutput:
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.original_stdout = sys.stdout
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        
    def __enter__(self):
        self.log_file = open(self.log_file_path, 'w')
        sys.stdout = self
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        if self.log_file:
            self.log_file.close()
            
    def write(self, message):
        self.original_stdout.write(message)
        if self.log_file:
            self.log_file.write(message)
            
    def flush(self):
        self.original_stdout.flush()
        if self.log_file:
            self.log_file.flush()


def compute_kl_divergence_kde(samples_p, samples_q, param_indices=None, max_samples_kde=50000):
    samples_p = samples_p.reshape(-1, 1) if samples_p.ndim == 1 else samples_p
    samples_q = samples_q.reshape(-1, 1) if samples_q.ndim == 1 else samples_q
    
    if param_indices is not None:
        samples_p, samples_q = samples_p[:, param_indices], samples_q[:, param_indices]
    
    def subsample(s, n): return s[np.random.choice(len(s), n, replace=False)] if len(s) > n else s
    samples_p_kde, samples_q_kde = subsample(samples_p, max_samples_kde), subsample(samples_q, max_samples_kde)
    
    per_param_kl = {}
    for i in range(samples_p.shape[1]):
        kde_p, kde_q = stats.gaussian_kde(samples_p_kde[:, i]), stats.gaussian_kde(samples_q_kde[:, i])
        x_min, x_max = min(samples_p[:, i].min(), samples_q[:, i].min()), max(samples_p[:, i].max(), samples_q[:, i].max())
        x_range = x_max - x_min
        x_grid = np.linspace(x_min - 0.1*x_range, x_max + 0.1*x_range, 1000)
        p_vals, q_vals = np.maximum(kde_p(x_grid), 1e-10), np.maximum(kde_q(x_grid), 1e-10)
        mask = p_vals > 1e-8
        integrand = np.where(mask, p_vals * np.log(p_vals / q_vals), 0)
        per_param_kl[i] = max(0.0, np.trapezoid(integrand, x_grid))
    
    return sum(per_param_kl.values()), per_param_kl


def load_montepython_chains(chain_dir, param_names, thin=1, scales=None):
    chain_files = sorted(Path(chain_dir).glob('*.txt'))
    if not chain_files:
        raise ValueError(f"No chain files found in {chain_dir}")
    
    print(f"Loading {len(chain_files)} MontePython chain files from {chain_dir}")
    samples_list, loglikes_list, n_params = [], [], len(param_names)
    scales_arr = np.ones(n_params) if scales is None else np.array(scales)
    
    for chain_file in chain_files:
        data = np.atleast_2d(np.loadtxt(chain_file))
        mult, neg_loglkl, params = data[:, 0].astype(int), data[:, 1], data[:, 2:2+n_params]
        # MontePython stores values in internal (unscaled) units; convert to physical units.
        params = params * scales_arr
        chain_samples = np.repeat(params, mult, axis=0)[::thin]
        chain_loglikes = -np.repeat(neg_loglkl, mult)[::thin]
        samples_list.append(chain_samples)
        loglikes_list.append(chain_loglikes)
    
    print(f"Loaded {sum(len(s) for s in samples_list)} total samples (thinned by {thin})")
    return samples_list, loglikes_list


def load_cobaya_chains(chain_dir, param_names, thin=1):
    chain_files = sorted(Path(chain_dir).glob('*.txt'))
    if not chain_files:
        raise ValueError(f"No chain files found in {chain_dir}")
    
    print(f"Loading {len(chain_files)} Cobaya chain files from {chain_dir}")
    samples_list, loglikes_list, n_params = [], [], len(param_names)
    
    for chain_file in chain_files:
        data = np.loadtxt(chain_file)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        with open(chain_file, 'r') as f:
            header = f.readline().strip('#').split()
        
        mult = data[:, header.index('weight')].astype(int)
        minuslogpost = data[:, header.index('minuslogpost')]

        loglkl = -minuslogpost
        
        param_cols = []
        for pname in param_names:
            if pname in header:
                param_cols.append(header.index(pname))
            else:
                raise ValueError(f"Parameter {pname} not found in chain header: {header}")
        
        params = data[:, param_cols]
        
        chain_samples = np.repeat(params, mult, axis=0)[::thin]
        chain_loglikes = np.repeat(loglkl, mult)[::thin]
        samples_list.append(chain_samples)
        loglikes_list.append(chain_loglikes)
    
    print(f"Loaded {sum(len(s) for s in samples_list)} total samples (thinned by {thin})")
    return samples_list, loglikes_list


def detect_chain_format(chain_dir):
    """Detect whether chains are MontePython or Cobaya format by checking for header."""
    chain_files = list(Path(chain_dir).glob('*.txt'))
    if not chain_files:
        raise ValueError(f"No chain files found in {chain_dir}")
    
    with open(chain_files[0], 'r') as f:
        first_line = f.readline().strip()
    
    if first_line.startswith('#'):
        return 'cobaya'
    else:
        return 'montepython'


def getdist_chain_inputs(chain, log_prob):
    """Convert sampler output into GetDist's list-of-independent-chains format."""
    chain = np.asarray(chain)
    log_prob = np.asarray(log_prob)

    if chain.ndim == 4:
        # AIES: (n_steps, n_chains, n_walkers, ndim)
        if log_prob.shape != chain.shape[:-1]:
            raise ValueError(
                f"log_prob shape {log_prob.shape} is incompatible with chain shape {chain.shape}"
            )
        samples = [
            chain[:, chain_idx, :, :].reshape(-1, chain.shape[-1])
            for chain_idx in range(chain.shape[1])
        ]
        loglikes = [
            -log_prob[:, chain_idx, :].reshape(-1)
            for chain_idx in range(chain.shape[1])
        ]
        return samples, loglikes

    if chain.ndim == 3:
        # Non-ensemble samplers: (n_steps, n_chains, ndim)
        if log_prob.shape != chain.shape[:-1]:
            raise ValueError(
                f"log_prob shape {log_prob.shape} is incompatible with chain shape {chain.shape}"
            )
        samples = [chain[:, chain_idx, :] for chain_idx in range(chain.shape[1])]
        loglikes = [-log_prob[:, chain_idx] for chain_idx in range(chain.shape[1])]
        return samples, loglikes

    if chain.ndim == 2:
        # Already flattened output from older cache files.
        if log_prob.shape != chain.shape[:1]:
            raise ValueError(
                f"log_prob shape {log_prob.shape} is incompatible with chain shape {chain.shape}"
            )
        return [chain], [-log_prob]

    raise ValueError(f"Unexpected chain shape {chain.shape}")


def print_diagnostics(samples, reference_samples, param_names, getdist_names, args, iteration, config_yaml, run_dir, surrogate=None, surrogate_sampler='ensemble', reference_sampler=None):
    print(f"=== BENCHMARK DIAGNOSTICS - ITERATION {iteration} ===")
    print(f"Configuration: {config_yaml}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Run directory: {run_dir}")
    print(f"Surrogate sampler: {surrogate_sampler}")
    if reference_sampler:
        print(f"Reference sampler: {reference_sampler}")
    print(f"Chain shape: {samples.samples.shape}")
    print(f"Thin factor: {args.thin}")
    print(f"MCMC steps: {args.n_steps}")
    if args.chains:
        print(f"Reference chains loaded from: {args.chains}")
    print("=" * 70)
    
    print(f"\n=== Convergence Diagnostics (Surrogate - {surrogate_sampler}) ===")
    try:
        surrogate_gr = samples.getGelmanRubin()
        print(f"Gelman-Rubin statistic: {surrogate_gr:.4f}")
        if surrogate_gr > 1.1:
            print("  WARNING: Gelman-Rubin > 1.1, chain may not be converged!")
        print("\nWARNING: Gelman-Rubin diagnostic is not reliable with the ensemble")
        print("         ensemble sampler as walkers are not independent chains.")
    except Exception as e:
        print(f"Gelman-Rubin statistic: N/A ({e})")
    
    print(f"\nEffective sample sizes:")
    for i, pname in enumerate(param_names):
        try:
            ess = samples.getEffectiveSamples(i)
            print(f"  {pname}: {ess:.0f}")
        except Exception as e:
            print(f"  {pname}: N/A ({e})")
    
    if reference_samples:
        ref_label = f"Reference ({reference_sampler})" if reference_sampler else "Reference/True Chains"
        print(f"\n=== Convergence Diagnostics ({ref_label}) ===")
        try:
            reference_gr = reference_samples.getGelmanRubin()
            print(f"Gelman-Rubin statistic: {reference_gr:.4f}")
            if reference_gr > 1.1:
                print("  WARNING: Gelman-Rubin > 1.1, chain may not be converged!")
        except Exception as e:
            print(f"Gelman-Rubin statistic: N/A ({e})")
        
        print(f"\nEffective sample sizes:")
        for i, pname in enumerate(param_names):
            try:
                ess = reference_samples.getEffectiveSamples(i)
                print(f"  {pname}: {ess:.0f}")
            except Exception as e:
                print(f"  {pname}: N/A ({e})")
    
    print(f"\n=== Posterior Statistics ===")
    surrogate_means = samples.getMeans()
    surrogate_stds = np.sqrt(samples.getVars())
    
    if reference_samples:
        reference_means = reference_samples.getMeans()
        reference_stds = np.sqrt(reference_samples.getVars())
        
        header = f"{'Parameter':<20} {'Surr Mean':>12} {'True Mean':>12} {'Mean Diff':>10} {'Rel (%)':>8} {'Surr Std':>10} {'True Std':>10} {'Std Diff':>10} {'Rel (%)':>8}"
        print(header)
        print("-" * len(header))
        
        for i, pname in enumerate(param_names):
            mean_diff = abs(surrogate_means[i] - reference_means[i])
            relative_mean_diff = mean_diff / abs(reference_means[i]) * 100
            std_diff = abs(surrogate_stds[i] - reference_stds[i])
            relative_std_diff = std_diff / abs(reference_stds[i]) * 100
            print(
                f"{pname:<20} "
                f"{surrogate_means[i]:>12.4f} {reference_means[i]:>12.4f} "
                f"{mean_diff:>10.4f} {relative_mean_diff:>8.1f} "
                f"{surrogate_stds[i]:>10.4f} {reference_stds[i]:>10.4f} "
                f"{std_diff:>10.4f} {relative_std_diff:>8.1f}"
            )
    else:
        print(f"{'Parameter':<20} {'Mean':>12} {'Std':>10}")
        print("-" * 45)
        for i, pname in enumerate(param_names):
            print(f"{pname:<20} {surrogate_means[i]:>12.4f} {surrogate_stds[i]:>10.4f}")
    
    if reference_samples:
        print(f"\n=== KL Divergence Analysis ===")
        print("Computing D_KL(True || Surrogate) for marginal distributions...")
        print("(measures information lost when using surrogate instead of true posterior)")
        
        try:
            print(f"\n{'Parameter':<20} {'KL (nats)':>15} {'KL (bits)':>15}")
            print("-" * 52)
            
            kl_values = []
            for i, pname in enumerate(param_names):
                kl_nats = compute_kl_divergence_kde(
                    reference_samples.samples[:, i:i+1],
                    samples.samples[:, i:i+1],
                    param_indices=[0],
                    max_samples_kde=5000
                )[1][0]
                kl_bits = kl_nats / np.log(2)
                kl_values.append(kl_nats)
                print(f"{pname:<20} {kl_nats:>15.6f} {kl_bits:>15.6f}")
            
            rms_kl = np.sqrt(np.mean(np.array(kl_values)**2))
            
            print(f"\nSummary:")
            print(f"  RMS KL divergence: {rms_kl:.6f} nats ({rms_kl/np.log(2):.6f} bits)")
            print(f"\nInterpretation:")
            print(f"  < 0.01 nats: Excellent agreement")
            print(f"  0.01-0.1 nats: Good agreement")
            print(f"  0.1-0.5 nats: Moderate discrepancy")
            print(f"  > 0.5 nats: Significant discrepancy")
            
        except Exception as e:
            print(f"Error computing KL divergence: {e}")
    
    print(f"\n=== Maximum Log-Likelihood Samples (from MCMC chains) ===")
    surrogate_bestfit = samples.samples[np.argmin(samples.loglikes)]
    print(f"Surrogate ({surrogate_sampler}) maximum of log(likelihood): {-min(samples.loglikes):.4f}")
    
    if reference_samples:
        reference_bestfit = reference_samples.samples[np.argmin(reference_samples.loglikes)]
        print(f"Reference ({reference_sampler}) maximum of log(likelihood): {-min(reference_samples.loglikes):.4f}")
        if surrogate is not None:
            surr_at_true_map = float(surrogate.logpost(tf.cast(reference_bestfit.reshape(1, -1), tf.float32)).numpy()[0])
            print(f"Surrogate log(likelihood) at reference ({reference_sampler}) best-fit: {surr_at_true_map:.4f}")
        
        print()
        header = f"{'Parameter':<20} {'Surr MAP':>12} {'True MAP':>12} {'Diff':>10} {'Rel (%)':>8}"
        print(header)
        print("-" * len(header))
        
        map_diffs = []
        for i, pname in enumerate(param_names):
            diff = abs(surrogate_bestfit[i] - reference_bestfit[i])
            rel_diff = diff / abs(reference_bestfit[i]) * 100
            map_diffs.append(diff)
            print(
                f"{pname:<20} "
                f"{surrogate_bestfit[i]:>12.4f} {reference_bestfit[i]:>12.4f} "
                f"{diff:>10.4f} {rel_diff:>8.1f}"
            )
        
        print(f"\nMAP difference RMS: {np.sqrt(np.mean(np.array(map_diffs)**2)):.4f}")
    else:
        print(f"{'Parameter':<20} {'Surrogate MAP':>15}")
        print("-" * 37)
        for i, pname in enumerate(param_names):
            print(f"{pname:<20} {surrogate_bestfit[i]:>15.4f}")
    
    print(f"\n=== 68% / 95% Credible Intervals ===")
    surrogate_stats = samples.getMargeStats()
    
    def fmt(L):
        if L is None: return "N/A"
        if getattr(L, "twotail", False): return f"[{L.lower:.4f}, {L.upper:.4f}]"
        if getattr(L, "onetail_lower", 0): return f"> {L.lower:.4f}"
        if getattr(L, "onetail_upper", 0): return f"< {L.upper:.4f}"
        return "N/A"
    
    lookup_names = getdist_names
    if reference_samples:
        reference_stats = reference_samples.getMargeStats()
        print(f"{'Parameter':<20} {'Surr 68%':<22} {'True 68%':<22} {'Surr 95%':<22} {'True 95%':<22}")
        print("-" * 110)
        
        for pname, sname in zip(param_names, lookup_names):
            ps = surrogate_stats.parWithName(sname)
            pt = reference_stats.parWithName(sname)
            s68, s95 = ps.limits[0] if ps is not None and ps.limits else None, ps.limits[1] if ps is not None and ps.limits else None
            t68, t95 = pt.limits[0] if pt is not None and pt.limits else None, pt.limits[1] if pt is not None and pt.limits else None
            print(f"{pname:<20} {fmt(s68):<22} {fmt(t68):<22} {fmt(s95):<22} {fmt(t95):<22}")
    else:
        print(f"{'Parameter':<20} {'68% Interval':<25} {'95% Interval':<25}")
        print("-" * 72)
        
        for pname, sname in zip(param_names, lookup_names):
            ps = surrogate_stats.parWithName(sname)
            s68, s95 = ps.limits[0] if ps is not None and ps.limits else None, ps.limits[1] if ps is not None and ps.limits else None
            print(f"{pname:<20} {fmt(s68):<25} {fmt(s95):<25}")

    print(f"\n=== END DIAGNOSTICS ===")


def main():
    parser = argparse.ArgumentParser(description='Benchmark surrogate likelihood')
    parser.add_argument('run_dir', help='Path to run directory')
    parser.add_argument('-i', '--iteration', type=int, default=None, help='Iteration to benchmark (auto-detects latest if not specified)')
    parser.add_argument('-n', '--n-steps', type=int, default=None, help='Number of MCMC steps (defaults to max_steps from config)')
    parser.add_argument('-t', '--thin', type=int, default=1, help='Thinning factor for chains')
    parser.add_argument('-p', '--params', nargs='+', default=None, help='Parameter indices to include in analysis')
    parser.add_argument('-c', '--chains', default=None, help='Path to MontePython or Cobaya chains directory for comparison')
    parser.add_argument('--no-training-data', action='store_true', help='Skip loading training data visualization')
    parser.add_argument('--no-training-history', action='store_true', help='Skip loading training history')
    args = parser.parse_args()
    
    timestamp, run_dir = datetime.now().strftime('%Y%m%d_%H%M%S'), Path(args.run_dir)
    
    yaml_files = list(run_dir.glob('*.yaml'))
    if not yaml_files:
        raise FileNotFoundError(f"No .yaml file found in {run_dir}")
    if len(yaml_files) > 1:
        print(f"Warning: Multiple .yaml files found in {run_dir}. Using {yaml_files[0].name}")
    config_yaml = yaml_files[0]
    
    with open(config_yaml) as f:
        config = yaml.safe_load(f)

    if args.n_steps is None:
        args.n_steps = config['sampling']['max_steps']
    
    if args.iteration is None:
        trained_models_dir = run_dir / 'trained_models'
        if not trained_models_dir.exists():
            raise FileNotFoundError(f"trained_models directory not found in {run_dir}")
        
        model_files = list(trained_models_dir.glob('trained_model_it_*.keras'))
        if not model_files:
            raise FileNotFoundError(f"No trained models found in {trained_models_dir}")
        
        iterations = []
        for f in model_files:
            try:
                it_num = int(f.stem.split('_')[-1])
                iterations.append(it_num)
            except ValueError:
                continue
        
        if not iterations:
            raise ValueError(f"Could not parse iteration numbers from model files in {trained_models_dir}")
        
        iteration = max(iterations)
        print(f"Auto-detected latest iteration: {iteration}")
    else:
        iteration = args.iteration
    metadata = SurrogateMetadata.load(run_dir / 'metadata.json')

    model = load_model(run_dir / f"trained_models/trained_model_it_{iteration}.keras")
    surrogate = SurrogateLikelihood(model, metadata)
    scales = metadata.scales

    param_names = surrogate.get_param_names()
    prior_bounds = surrogate.get_prior_bounds()
    param_labels = surrogate.get_param_labels()

    getdist_names = [
        re.sub(r'[\s*?]', '', name)
        for name in param_names
    ]

    getdist_ranges = {
        getdist_name: prior_bounds[param_name]
        for param_name, getdist_name in zip(param_names, getdist_names)
    }

    x_all = None
    if not args.no_training_data:
        data_path = run_dir / "training_data" / f'training_data_it_{iteration}.csv'
        if data_path.exists():
            df = pd.read_csv(data_path)
            x_all = df[param_names].to_numpy()

    burn_in = config['sampling']['burn_in']
    n_walkers = config['sampling']['n_walkers']
    n_chains = config['sampling']['n_chains']
    sampler_name = config['sampling']['sampler']

    output_dir = run_dir / 'benchmark_chains'
    output_dir.mkdir(exist_ok=True)
    chain_path = output_dir / (
        f'benchmark_chain_it_{iteration}_{sampler_name}_'
        f'steps{args.n_steps}_thin{args.thin}.npz'
    )

    if chain_path.exists():
        data = np.load(chain_path)
        chain, log_prob = data['chain'], data['log_prob']
        print(f"Loaded chain: {chain.shape}")
    else:
        print(f"Running {sampler_name} sampler: {n_walkers} walkers, {burn_in} burn-in, {args.n_steps} steps")
        lower = [b[0] for b in prior_bounds.values()]
        upper = [b[1] for b in prior_bounds.values()]
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
            size=(n_chains, n_walkers, len(param_names))
        )
        sampler.run(initial_pos=initial_pos, max_steps=args.n_steps)
        chain = sampler.get_chain(discard=burn_in, thin=args.thin)
        log_prob = sampler.get_logpost(discard=burn_in, thin=args.thin)
        np.savez(chain_path, chain=chain, log_prob=log_prob)
        print(f"Chain shape: {chain.shape}")

    getdist_samples, getdist_loglikes = getdist_chain_inputs(chain, log_prob)
    samples = MCSamples(samples=getdist_samples,
                        names=getdist_names, labels=param_labels,
                        loglikes=getdist_loglikes,
                        ranges=getdist_ranges)
    
    if args.params:
        if len(args.params) == 1 and ',' in args.params[0]:
            param_indices = [int(x) - 1 for x in args.params[0].split(',')]
            plot_params = [getdist_names[i] for i in param_indices]
        else:
            param_indices = [param_names.index(p) for p in args.params]
            plot_params = [getdist_names[i] for i in param_indices]
    else:
        plot_params, param_indices = getdist_names, list(range(len(param_names)))
    
    reference_samples = None
    chain_format = None
    if args.chains:
        try:
            chain_format = detect_chain_format(args.chains)
            print(f"Detected {chain_format} chain format")
            
            if chain_format == 'cobaya':
                reference_samples_list, reference_loglkls_list = load_cobaya_chains(args.chains, param_names, thin=1)
            else:
                reference_samples_list, reference_loglkls_list = load_montepython_chains(
                    args.chains, param_names, thin=1, scales=scales)
            
            reference_samples = MCSamples(samples=reference_samples_list, names=getdist_names, labels=param_labels,
                                         loglikes=[-ll for ll in reference_loglkls_list], ranges=getdist_ranges)
            print(f"Loaded {sum(len(s) for s in reference_samples_list)} samples from {chain_format} chains")
        except Exception as e:
            print(f"Warning: Could not load chains: {e}")
    
    surrogate_sampler = config['sampling']['sampler']
    reference_sampler = None
    if args.chains and chain_format:
        reference_sampler = chain_format

    log_file_path = run_dir / "benchmark_results" / f"{timestamp}_diagnostics_it_{iteration}.log"
    log_file_path.parent.mkdir(exist_ok=True)
    
    with TeeOutput(str(log_file_path)):
        print_diagnostics(samples, reference_samples, param_names, getdist_names, args, iteration, config_yaml, run_dir, surrogate,
                         surrogate_sampler=surrogate_sampler, reference_sampler=reference_sampler)
    
    g = plots.get_subplot_plotter(width_inch=width_inches)
    g.settings.axes_fontsize = fontsize
    g.settings.axes_labelsize = fontsize
    g.settings.legend_fontsize = fontsize * 0.9
    g.settings.figure_legend_frame = False
    
    plot_data = ([reference_samples, samples] if reference_samples else samples)
    plot_args = ({"filled": False, "param_limits": {n: getdist_ranges[n] for n in plot_params}})
    
    if reference_samples:
        plot_args.update({"line_args": [{"lw": 2, "color": "C1"}, {"lw": 2, "color": "C0"}],
                         "contour_args": [{"lw": 2, "color": "C1"}, {"lw": 2, "color": "C0"}]})
    else:
        plot_args.update({"line_args": [{"lw": 2, "color": "C0"}],
                         "contour_args": [{"lw": 2, "color": "C0"}]})
    
    g.triangle_plot(plot_data, plot_params, **plot_args)
    
    if x_all is not None:
        # Keep dot size a fixed fraction of panel width regardless of n_params.
        # s is in points²; panel_pts ≈ figure_width_pts / n_plot, so s ∝ 1/n_plot².
        n_plot = len(plot_params)
        panel_pts = width_inches * 72 / n_plot
        s = (0.01 * panel_pts) ** 2  # dot radius ≈ 0.25% of panel width
        for j in range(1, len(plot_params)):
            for i in range(j):
                if ax := g.get_axes_for_params(plot_params[i], plot_params[j]):
                    ax.scatter(x_all[:, param_indices[i]], x_all[:, param_indices[j]],
                              s=s, alpha=0.15, color='black', zorder=1, edgecolors='none', rasterized=True)

    [legend.remove() for legend in g.fig.legends]
    
    ref_label = f'Reference ({reference_sampler})' if reference_sampler else 'True Posterior'
    surr_label = f'Surrogate ({surrogate_sampler})'
    
    legend_elements = [Line2D([0], [0], color='C0', lw=2, label=surr_label)]
    if reference_samples:
        legend_elements.append(Line2D([0], [0], color='C1', lw=2, label=ref_label))
    if x_all is not None:
        legend_elements.append(Line2D([0], [0], marker='o', color='black', lw=0, ms=4, alpha=0.7, label='Training Data'))
    
    g.fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98), 
                fontsize=fontsize * 0.9, framealpha=0.9)
    
    figure_dir = run_dir / 'benchmark_figures'
    figure_dir.mkdir(exist_ok=True)
    output_path = figure_dir / f'{timestamp}_triangle_plot_it_{iteration}.pdf'
    g.export(str(output_path))
    
    if not args.no_training_history and (history_path := run_dir / f"training_history/history_it_{iteration}.csv").exists():
        history = pd.read_csv(history_path)

        fig, ax = plt.subplots(figsize=(width_inches, width_inches * 0.6))
        epochs = range(len(history['loss']))
        ax.plot(epochs, history['loss'].to_numpy(), label='Training', color='blue', alpha=0.8)
        ax.plot(epochs, history['val_loss'].to_numpy(), label='Validation', color='orange', alpha=0.8)
        ax.set(xlabel='Epoch', ylabel='Loss', title=f'Training History - Iteration {iteration}', yscale='log')
        ax.grid(alpha=0.3, linewidth=0.5)
        ax.legend()

        fig.subplots_adjust(left=0.15, right=0.97, bottom=0.15, top=0.92)
        
        history_output = figure_dir / f'{timestamp}_training_history_it_{iteration}.pdf'
        fig.savefig(history_output, format='pdf')
        plt.close(fig)
    
    print(f"\n=== Benchmark Complete ===\nChain: {chain_path}\nPlots: {figure_dir}\nLog: {log_file_path}")


if __name__ == '__main__':
    main()
