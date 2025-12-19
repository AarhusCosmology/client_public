# benchmarking/benchmark.py

import os
import sys
import argparse
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
from datetime import datetime
from getdist import MCSamples, plots
from matplotlib.lines import Line2D
from scipy import stats
from config.config_loader import load_config
from likelihood.surrogate import EmulatedLikelihood
from likelihood.montepython_wrapper import MontePythonLikelihood
from likelihood.cobaya_wrapper import CobayaLikelihood
from sampling.sampler import run_sampler
from training.training import load_training_data

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


def load_likelihood_from_config(cfg):
    if cfg.wrapper == 'montepython':
        return MontePythonLikelihood(cfg.param, cfg.conf, 'resources/montepython_public/montepython', silent=True)
    elif cfg.wrapper == 'cobaya':
        return CobayaLikelihood(cfg.param, debug=False)
    raise ValueError(f"Unknown likelihood wrapper: {cfg.wrapper}")


def load_montepython_chains(chain_dir, param_names, thin=1):
    chain_files = sorted(Path(chain_dir).glob('*.txt'))
    if not chain_files:
        raise ValueError(f"No chain files found in {chain_dir}")
    
    print(f"Loading {len(chain_files)} MontePython chain files from {chain_dir}")
    samples_list, loglikes_list, n_params = [], [], len(param_names)
    
    for chain_file in chain_files:
        data = np.atleast_2d(np.loadtxt(chain_file))
        mult, neg_loglkl, params = data[:, 0].astype(int), data[:, 1], data[:, 2:2+n_params]
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
        minuslogprior = data[:, header.index('minuslogprior')]
        
        loglkl = -(minuslogpost - minuslogprior)
        
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


def print_diagnostics(samples, mp_samples, param_names, args, iteration, config_yaml, run_dir, surrogate=None, surrogate_sampler='emcee', reference_sampler=None):
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
        surrogate_GR = samples.getGelmanRubin()
        print(f"Gelman-Rubin statistic: {surrogate_GR:.4f}")
        if surrogate_GR > 1.1:
            print("  WARNING: Gelman-Rubin > 1.1, chain may not be converged!")
        print("\nWARNING: Gelman-Rubin diagnostic is not reliable with the emcee")
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
    
    if mp_samples:
        ref_label = f"Reference ({reference_sampler})" if reference_sampler else "Reference/True Chains"
        print(f"\n=== Convergence Diagnostics ({ref_label}) ===")
        try:
            mp_GR = mp_samples.getGelmanRubin()
            print(f"Gelman-Rubin statistic: {mp_GR:.4f}")
            if mp_GR > 1.1:
                print("  WARNING: Gelman-Rubin > 1.1, chain may not be converged!")
        except Exception as e:
            print(f"Gelman-Rubin statistic: N/A ({e})")
        
        print(f"\nEffective sample sizes:")
        for i, pname in enumerate(param_names):
            try:
                ess = mp_samples.getEffectiveSamples(i)
                print(f"  {pname}: {ess:.0f}")
            except Exception as e:
                print(f"  {pname}: N/A ({e})")
    
    print(f"\n=== Posterior Statistics ===")
    surrogate_means = samples.getMeans()
    surrogate_stds = np.sqrt(samples.getVars())
    
    if mp_samples:
        mp_means = mp_samples.getMeans()
        mp_stds = np.sqrt(mp_samples.getVars())
        
        header = f"{'Parameter':<20} {'Surr Mean':>12} {'True Mean':>12} {'Mean Diff':>10} {'Rel (%)':>8} {'Surr Std':>10} {'True Std':>10} {'Std Diff':>10} {'Rel (%)':>8}"
        print(header)
        print("-" * len(header))
        
        for i, pname in enumerate(param_names):
            mean_diff = abs(surrogate_means[i] - mp_means[i])
            relative_mean_diff = mean_diff / abs(mp_means[i]) * 100
            std_diff = abs(surrogate_stds[i] - mp_stds[i])
            relative_std_diff = std_diff / abs(mp_stds[i]) * 100
            print(
                f"{pname:<20} "
                f"{surrogate_means[i]:>12.4f} {mp_means[i]:>12.4f} "
                f"{mean_diff:>10.4f} {relative_mean_diff:>8.1f} "
                f"{surrogate_stds[i]:>10.4f} {mp_stds[i]:>10.4f} "
                f"{std_diff:>10.4f} {relative_std_diff:>8.1f}"
            )
    else:
        print(f"{'Parameter':<20} {'Mean':>12} {'Std':>10}")
        print("-" * 45)
        for i, pname in enumerate(param_names):
            print(f"{pname:<20} {surrogate_means[i]:>12.4f} {surrogate_stds[i]:>10.4f}")
    
    if mp_samples:
        print(f"\n=== KL Divergence Analysis ===")
        print("Computing D_KL(True || Surrogate) for marginal distributions...")
        print("(measures information lost when using surrogate instead of true posterior)")
        
        try:
            print(f"\n{'Parameter':<20} {'KL (nats)':>15} {'KL (bits)':>15}")
            print("-" * 52)
            
            kl_values = []
            for i, pname in enumerate(param_names):
                kl_nats = compute_kl_divergence_kde(
                    mp_samples.samples[:, i:i+1],
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
    
    if mp_samples:
        mp_bestfit = mp_samples.samples[np.argmin(mp_samples.loglikes)]
        print(f"Reference ({reference_sampler}) maximum of log(likelihood): {-min(mp_samples.loglikes):.4f}")
        if surrogate is not None:
            surr_at_true_map = surrogate.logpost_array(mp_bestfit.reshape(1, -1))[0]
            print(f"Surrogate log(likelihood) at reference ({reference_sampler}) best-fit: {surr_at_true_map:.4f}")
        
        print()
        header = f"{'Parameter':<20} {'Surr MAP':>12} {'True MAP':>12} {'Diff':>10} {'Rel (%)':>8}"
        print(header)
        print("-" * len(header))
        
        map_diffs = []
        for i, pname in enumerate(param_names):
            diff = abs(surrogate_bestfit[i] - mp_bestfit[i])
            rel_diff = diff / abs(mp_bestfit[i]) * 100
            map_diffs.append(diff)
            print(
                f"{pname:<20} "
                f"{surrogate_bestfit[i]:>12.4f} {mp_bestfit[i]:>12.4f} "
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
    
    if mp_samples:
        mp_stats = mp_samples.getMargeStats()
        print(f"{'Parameter':<20} {'Surr 68%':<22} {'True 68%':<22} {'Surr 95%':<22} {'True 95%':<22}")
        print("-" * 110)
        
        for pname in param_names:
            ps = surrogate_stats.parWithName(pname)
            pt = mp_stats.parWithName(pname)
            s68, s95 = ps.limits[0] if ps.limits else None, ps.limits[1] if ps.limits else None
            t68, t95 = pt.limits[0] if pt.limits else None, pt.limits[1] if pt.limits else None
            print(f"{pname:<20} {fmt(s68):<22} {fmt(t68):<22} {fmt(s95):<22} {fmt(t95):<22}")
    else:
        print(f"{'Parameter':<20} {'68% Interval':<25} {'95% Interval':<25}")
        print("-" * 72)
        
        for pname in param_names:
            ps = surrogate_stats.parWithName(pname)
            s68, s95 = ps.limits[0] if ps.limits else None, ps.limits[1] if ps.limits else None
            print(f"{pname:<20} {fmt(s68):<25} {fmt(s95):<25}")
    
    print(f"\n=== END DIAGNOSTICS ===")


def main():
    parser = argparse.ArgumentParser(description='Benchmark surrogate likelihood')
    parser.add_argument('run_dir', help='Path to run directory')
    parser.add_argument('-it', '--iteration', type=int, default=None, help='Iteration to benchmark (auto-detects latest if not specified)')
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
    
    cfg = load_config(str(config_yaml))
    
    if args.n_steps is None:
        args.n_steps = cfg.max_steps
    
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
    true_likelihood = load_likelihood_from_config(cfg)
    param_names = true_likelihood.varying_param_names
    
    if getattr(cfg, 'n_std', None):
        true_likelihood.restrict_prior(n_std=cfg.n_std)
    prior_bounds = true_likelihood.get_prior_bounds()
    
    x_all = None
    if not args.no_training_data:
        training_data_dir = run_dir / "training_data"
        x_list = [load_training_data(str(f))[0] for i in range(iteration + 1) 
                  if (f := training_data_dir / f'data_it_{i}.h5').exists()]
        x_all = np.vstack(x_list) if x_list else None
    
    surrogate = EmulatedLikelihood(
        str(run_dir / f"trained_models/trained_model_it_{iteration}.keras"),
        str(run_dir / f"scalers/x_scaler_it_{iteration}.pkl"),
        str(run_dir / f"scalers/y_scaler_it_{iteration}.pkl"),
        true_likelihood=true_likelihood
    )
    
    output_dir = run_dir / 'benchmark_chains'
    output_dir.mkdir(exist_ok=True)
    chain_path = output_dir / f'benchmark_chain_it_{iteration}.h5'
    
    if chain_path.exists():
        import emcee
        reader = emcee.backends.HDFBackend(str(chain_path))
        chain, log_prob = reader.get_chain(thin=args.thin, discard=cfg.burn_in), reader.get_log_prob(thin=args.thin, discard=cfg.burn_in)
        print(f"Loaded chain: {chain.shape}")
    else:
        print(f"Running emcee: {cfg.n_walkers} walkers, {cfg.burn_in} burn-in, {args.n_steps} steps")
        chain, log_prob, _ = run_sampler(cfg, surrogate, str(chain_path), vectorize=True, flat=False, 
                                         temper=False, n_steps=args.n_steps, return_metrics=True)
        if args.thin > 1:
            chain, log_prob = chain[::args.thin], log_prob[::args.thin]
    
    param_labels = [true_likelihood.param['varying'][p].get('label', p).replace('$', '') for p in param_names]
    
    label_overrides = {
        'deg_ncdm__2': r'N_{\mathrm{eff},s}',
        'm_ncdm__2': r'm_s'
    }
    for i, pname in enumerate(param_names):
        if pname in label_overrides:
            param_labels[i] = label_overrides[pname]
    
    samples = MCSamples(samples=[chain[:, i, :] for i in range(chain.shape[1])], 
                        names=param_names, labels=param_labels,
                        loglikes=[-log_prob[:, i] for i in range(log_prob.shape[1])], 
                        ranges=prior_bounds)
    
    if args.params:
        if len(args.params) == 1 and ',' in args.params[0]:
            param_indices = [int(x) - 1 for x in args.params[0].split(',')]
            plot_params = [param_names[i] for i in param_indices]
        else:
            plot_params = args.params
            param_indices = [param_names.index(p) for p in plot_params]
    else:
        plot_params, param_indices = param_names, list(range(len(param_names)))
    
    mp_samples = None
    chain_format = None
    if args.chains:
        try:
            chain_format = detect_chain_format(args.chains)
            print(f"Detected {chain_format} chain format")
            
            if chain_format == 'cobaya':
                mp_samples_list, mp_loglikes_list = load_cobaya_chains(args.chains, param_names, thin=1)
            else:
                mp_samples_list, mp_loglikes_list = load_montepython_chains(args.chains, param_names, thin=1)
            
            mp_samples = MCSamples(samples=mp_samples_list, names=param_names, labels=param_labels,
                                  loglikes=[-ll for ll in mp_loglikes_list], ranges=prior_bounds)
            print(f"Loaded {sum(len(s) for s in mp_samples_list)} samples from {chain_format} chains")
        except Exception as e:
            print(f"Warning: Could not load chains: {e}")
    
    surrogate_sampler = cfg.sampling_method if hasattr(cfg, 'sampling_method') else 'emcee'
    reference_sampler = None
    if args.chains and chain_format:
        reference_sampler = chain_format
    
    log_file_path = run_dir / "benchmark_results" / f"{timestamp}_diagnostics_it_{iteration}.log"
    log_file_path.parent.mkdir(exist_ok=True), surrogate
    
    with TeeOutput(str(log_file_path)):
        print_diagnostics(samples, mp_samples, param_names, args, iteration, config_yaml, run_dir, surrogate, 
                         surrogate_sampler=surrogate_sampler, reference_sampler=reference_sampler)
    
    g = plots.get_subplot_plotter(width_inch=width_inches)
    g.settings.axes_fontsize = fontsize
    g.settings.axes_labelsize = fontsize
    g.settings.legend_fontsize = fontsize * 0.9
    g.settings.figure_legend_frame = False
    
    plot_data = ([mp_samples, samples] if mp_samples else samples)
    plot_args = {"filled": False, "param_limits": {n: prior_bounds[n] for n in plot_params}}
    
    if mp_samples:
        plot_args.update({"line_args": [{"lw": 2, "color": "C1"}, {"lw": 2, "color": "C0"}],
                         "contour_args": [{"lw": 2, "color": "C1"}, {"lw": 2, "color": "C0"}]})
    else:
        plot_args.update({"line_args": [{"lw": 2, "color": "C0"}],
                         "contour_args": [{"lw": 2, "color": "C0"}]})
    
    g.triangle_plot(plot_data, plot_params, **plot_args)
    
    if x_all is not None:
        for j in range(1, len(plot_params)):
            for i in range(j):
                if ax := g.get_axes_for_params(plot_params[i], plot_params[j]):
                    ax.scatter(x_all[:, param_indices[i]], x_all[:, param_indices[j]], 
                              s=0.3, alpha=0.15, color='black', zorder=1, edgecolors='none', rasterized=True)
    
    [legend.remove() for legend in g.fig.legends]
    
    ref_label = f'Reference ({reference_sampler})' if reference_sampler else 'True Posterior'
    surr_label = f'Surrogate ({surrogate_sampler})'
    
    legend_elements = [Line2D([0], [0], color='C0', lw=2, label=surr_label)]
    if mp_samples:
        legend_elements.append(Line2D([0], [0], color='C1', lw=2, label=ref_label))
    if x_all is not None:
        legend_elements.append(Line2D([0], [0], marker='o', color='black', lw=0, ms=4, alpha=0.7, label='Training Data'))
    
    g.fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98), 
                fontsize=fontsize * 0.9, framealpha=0.9)
    
    figure_dir = run_dir / 'benchmark_figures'
    figure_dir.mkdir(exist_ok=True)
    output_path = figure_dir / f'{timestamp}_triangle_plot_it_{iteration}.pdf'
    g.export(str(output_path))
    
    if not args.no_training_history and (history_path := run_dir / f"training_history/history_it_{iteration}.pkl").exists():
        with open(history_path, 'rb') as f:
            history = pickle.load(f)
        
        fig, ax = plt.subplots(figsize=(width_inches, width_inches * 0.6))
        epochs = range(len(history['loss']))
        ax.plot(epochs, history['loss'], label='Training', color='blue', alpha=0.8)
        ax.plot(epochs, history['val_loss'], label='Validation', color='orange', alpha=0.8)
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
