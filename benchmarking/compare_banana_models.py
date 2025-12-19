#!/usr/bin/env python3
"""
Compare iteration 9 models from two banana runs against the true banana likelihood.
This script evaluates both surrogate models and the true likelihood on a set of test points,
then computes various error metrics and creates visualizations.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import pickle

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
})

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from likelihood.base import BaseLikelihood
from likelihood.banana_function import banana_loglkl_planck
from likelihood.surrogate import EmulatedLikelihood
from config.config_loader import load_config


class BananaLikelihood(BaseLikelihood):
    """Direct wrapper for banana likelihood - no Cobaya overhead."""
    
    def __init__(self):
        super().__init__()
        self._setup_parameters()
    
    def _setup_parameters(self):
        """Setup parameter definitions from banana_planck.yaml."""
        param_defs = {
            'omega_b': {'range': [0.005, 0.1], 'initial': 0.02281, 'sigma': 0.0001773, 'label': r'\Omega_b h^2'},
            'omega_cdm': {'range': [0.001, 0.99], 'initial': 0.118, 'sigma': 0.0008154, 'label': r'\Omega_{cdm} h^2'},
            'theta_s_100': {'range': [0.5, 10], 'initial': 1.041, 'sigma': 0.0002518, 'label': r'100\,\theta_\mathrm{s}'},
            'ln10_10_A_s': {'range': [1.61, 3.91], 'initial': 3.009, 'sigma': 0.01124, 'label': r'\ln(10^{10}A_s)'},
            'n_s': {'range': [0.8, 1.2], 'initial': 0.9586, 'sigma': 0.004295, 'label': r'n_s'},
            'tau_reio': {'range': [0.01, 0.8], 'initial': 0.05981, 'sigma': 0.003923, 'label': r'\tau_\mathrm{reio}'},
            'A_cib_217': {'range': [0.0, 200.0], 'initial': 33.29, 'sigma': 8.93, 'label': r'A^\mathrm{CIB}_{217}'},
            'xi_sz_cib': {'range': [0.0, 1.0], 'initial': 0.3325, 'sigma': 0.242, 'label': r'\xi_{\mathrm{tSZ}\times\mathrm{CIB}}'},
            'A_sz': {'range': [0.0, 10.0], 'initial': 5.377, 'sigma': 2.097, 'label': r'A_\mathrm{tSZ}'},
            'ps_A_100_100': {'range': [0.0, 400.0], 'initial': 173.8, 'sigma': 26.43, 'label': r'A^{\mathrm{PS}}_{100\times100}'},
            'ps_A_143_143': {'range': [0.0, 400.0], 'initial': 57.1, 'sigma': 10.07, 'label': r'A^{\mathrm{PS}}_{143\times143}'},
            'ps_A_143_217': {'range': [0.0, 400.0], 'initial': 29.52, 'sigma': 5.415, 'label': r'A^{\mathrm{PS}}_{143\times217}'},
            'ps_A_217_217': {'range': [0.0, 400.0], 'initial': 127.1, 'sigma': 9.465, 'label': r'A^{\mathrm{PS}}_{217\times217}'},
            'ksz_norm': {'range': [0.0, 10.0], 'initial': 4.046, 'sigma': 2.234, 'label': r'A_\mathrm{kSZ}'},
            'gal545_A_100': {'range': [0.0, 50.0], 'initial': 5.413, 'sigma': 2.238, 'label': r'A^{\mathrm{gal},545}_{100}'},
            'gal545_A_143': {'range': [0.0, 50.0], 'initial': 10.89, 'sigma': 1.394, 'label': r'A^{\mathrm{gal},545}_{143}'},
            'gal545_A_143_217': {'range': [0.0, 100.0], 'initial': 24.27, 'sigma': 3.168, 'label': r'A^{\mathrm{gal},545}_{143\times217}'},
            'gal545_A_217': {'range': [0.0, 400.0], 'initial': 91.74, 'sigma': 6.017, 'label': r'A^{\mathrm{gal},545}_{217}'},
            'galf_TE_A_100': {'range': [0.0, 10.0], 'initial': 0.01359, 'sigma': 0.03845, 'label': r'A^{\mathrm{gal},TE}_{100}'},
            'galf_TE_A_100_143': {'range': [0.0, 10.0], 'initial': 0.142, 'sigma': 0.03303, 'label': r'A^{\mathrm{gal},TE}_{100\times143}'},
            'galf_TE_A_100_217': {'range': [0.0, 10.0], 'initial': 0.5247, 'sigma': 0.06606, 'label': r'A^{\mathrm{gal},TE}_{100\times217}'},
            'galf_TE_A_143': {'range': [0.0, 10.0], 'initial': 0.2598, 'sigma': 0.06623, 'label': r'A^{\mathrm{gal},TE}_{143}'},
            'galf_TE_A_143_217': {'range': [0.0, 10.0], 'initial': 0.8805, 'sigma': 0.09459, 'label': r'A^{\mathrm{gal},TE}_{143\times217}'},
            'galf_TE_A_217': {'range': [0.0, 10.0], 'initial': 2.113, 'sigma': 0.2954, 'label': r'A^{\mathrm{gal},TE}_{217}'},
            'calib_100T': {'range': [0.0, 3.0], 'initial': 1.001, 'sigma': 0.0007192, 'label': r'c^{T}_{100}'},
            'calib_217T': {'range': [0.0, 3.0], 'initial': 0.997, 'sigma': 0.0005374, 'label': r'c^{T}_{217}'},
            'A_planck': {'range': [0.9, 1.1], 'initial': 0.9956, 'sigma': 0.002401, 'label': r'A_\mathrm{Planck}'},
            'x_28': {'range': [0.0, 5.0], 'initial': 2.5, 'sigma': 1.0, 'label': r'x_{28}'},
            'x_29': {'range': [0.0, 5.0], 'initial': 2.5, 'sigma': 1.0, 'label': r'x_{29}'},
        }
        
        self.param['varying'] = param_defs
    
    def _loglkl(self, position: dict) -> float:
        """Call banana likelihood function directly."""
        return banana_loglkl_planck(**position)
    
    def logprior(self, position: dict) -> float:
        """Uniform prior within bounds."""
        return self.log_uniform_prior(position)
    
    def get_parameter_info(self) -> dict:
        """Return parameter info."""
        return self.param


def load_model_and_likelihood(run_dir, iteration=9):
    """Load surrogate model and true likelihood for a given run."""
    run_path = Path(run_dir)
    
    yaml_files = list(run_path.glob('*.yaml'))
    if not yaml_files:
        raise FileNotFoundError(f"No YAML config found in {run_dir}")
    config_yaml = yaml_files[0]
    
    cfg = load_config(str(config_yaml))
    
    true_likelihood = BananaLikelihood()
    if hasattr(cfg, 'n_std') and cfg.n_std is not None:
        true_likelihood.restrict_prior(n_std=cfg.n_std)
    
    model_path = run_path / f"trained_models/trained_model_it_{iteration}.keras"
    x_scaler_path = run_path / f"scalers/x_scaler_it_{iteration}.pkl"
    y_scaler_path = run_path / f"scalers/y_scaler_it_{iteration}.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    surrogate = EmulatedLikelihood(
        str(model_path),
        str(x_scaler_path),
        str(y_scaler_path),
        true_likelihood=true_likelihood
    )
    
    return surrogate, true_likelihood, cfg


def load_chain_samples(chain_dir='banana_planck', max_samples=None, seed=42):
    """Load samples from chain files in the specified directory.
    
    Args:
        chain_dir: Directory containing chain files (*.txt)
        max_samples: Maximum number of samples to load (None = all)
        seed: Random seed for subsampling
    
    Returns:
        samples: Array of shape (n_samples, n_params) with parameter values
    """
    chain_path = Path(chain_dir)
    chain_files = sorted(chain_path.glob('*.txt'))
    
    if not chain_files:
        raise FileNotFoundError(f"No chain files found in {chain_dir}")
    
    print(f"Loading chain samples from {len(chain_files)} files in {chain_dir}...")
    
    all_samples = []
    for chain_file in chain_files:
        if not chain_file.stem.isdigit():
            continue
        
        data = np.loadtxt(chain_file, comments='#')
        params = data[:, 2:31]
        all_samples.append(params)
    
    all_samples = np.vstack(all_samples)
    print(f"  Loaded {len(all_samples)} samples total")
    
    if max_samples is not None and len(all_samples) > max_samples:
        np.random.seed(seed)
        indices = np.random.choice(len(all_samples), max_samples, replace=False)
        all_samples = all_samples[indices]
        print(f"  Subsampled to {len(all_samples)} samples")
    
    return all_samples


def generate_test_points(likelihood, n_samples=1000, strategy='lhs', seed=42):
    """Generate test points from the prior."""
    np.random.seed(seed)
    
    param_names = likelihood.varying_param_names
    n_params = len(param_names)
    prior_bounds = likelihood.get_prior_bounds()
    
    if strategy == 'lhs':
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=n_params, seed=seed)
        unit_samples = sampler.random(n=n_samples)
        
        samples = np.zeros_like(unit_samples)
        for i, param_name in enumerate(param_names):
            lower, upper = prior_bounds[param_name]
            samples[:, i] = lower + unit_samples[:, i] * (upper - lower)
    
    elif strategy == 'grid':
        n_per_dim = int(np.ceil(n_samples ** (1.0 / n_params)))
        samples = np.zeros((n_per_dim ** n_params, n_params))
        
        grids = []
        for param_name in param_names:
            lower, upper = prior_bounds[param_name]
            grids.append(np.linspace(lower, upper, n_per_dim))
        
        mesh = np.meshgrid(*grids)
        for i in range(n_params):
            samples[:, i] = mesh[i].ravel()
        
        if len(samples) > n_samples:
            indices = np.random.choice(len(samples), n_samples, replace=False)
            samples = samples[indices]
    
    else:
        samples = np.zeros((n_samples, n_params))
        for i, param_name in enumerate(param_names):
            lower, upper = prior_bounds[param_name]
            samples[:, i] = np.random.uniform(lower, upper, n_samples)
    
    return samples


def evaluate_models(samples, true_likelihood, surrogate1, surrogate2, use_parallel=False):
    """Evaluate true likelihood and both surrogates on test points."""
    n_samples = len(samples)
    
    print(f"Evaluating {n_samples} test points...")
    
    print("  Evaluating true likelihood...")
    if use_parallel:
        from utils.mpi_utils import is_mpi_available, parallel_evaluate_likelihood
        param_names = true_likelihood.varying_param_names
        likelihood_func = lambda x: true_likelihood.loglkl({name: float(x[j]) for j, name in enumerate(param_names)})
        true_loglkls = parallel_evaluate_likelihood(samples, likelihood_func, description="true likelihood")
    else:
        param_names = true_likelihood.varying_param_names
        true_loglkls = np.array([
            true_likelihood.loglkl({name: float(samples[i, j]) for j, name in enumerate(param_names)})
            for i in range(n_samples)
        ])
    
    print("  Evaluating surrogate 1...")
    surr1_loglkls = surrogate1.predict(samples)
    
    print("  Evaluating surrogate 2...")
    surr2_loglkls = surrogate2.predict(samples)
    
    return true_loglkls, surr1_loglkls, surr2_loglkls


def compute_errors(true_vals, pred_vals):
    """Compute various error metrics."""
    abs_errors = np.abs(pred_vals - true_vals)
    squared_errors = (pred_vals - true_vals) ** 2
    
    finite_mask = np.isfinite(true_vals) & np.isfinite(pred_vals)
    
    if not np.any(finite_mask):
        return {
            'mae': np.nan,
            'rmse': np.nan,
            'max_abs_error': np.nan,
            'median_abs_error': np.nan,
            'relative_errors': np.array([]),
            'abs_errors': np.array([])
        }
    
    true_finite = true_vals[finite_mask]
    pred_finite = pred_vals[finite_mask]
    abs_errors_finite = abs_errors[finite_mask]
    squared_errors_finite = squared_errors[finite_mask]
    
    relative_errors = np.where(
        np.abs(true_finite) > 1e-10,
        abs_errors_finite / np.abs(true_finite),
        np.nan
    )
    
    metrics = {
        'mae': np.mean(abs_errors_finite),
        'rmse': np.sqrt(np.mean(squared_errors_finite)),
        'max_abs_error': np.max(abs_errors_finite),
        'median_abs_error': np.median(abs_errors_finite),
        'mean_relative_error': np.nanmean(relative_errors),
        'median_relative_error': np.nanmedian(relative_errors),
        'abs_errors': abs_errors_finite,
        'relative_errors': relative_errors[np.isfinite(relative_errors)]
    }
    
    return metrics


def print_comparison_table(errors1, errors2, run1_name, run2_name):
    """Print a comparison table of error metrics."""
    print("\n" + "="*80)
    print("ERROR METRICS COMPARISON")
    print("="*80)
    print(f"\n{'Metric':<30} {run1_name:<20} {run2_name:<20}")
    print("-"*80)
    
    metrics = [
        ('Mean Absolute Error', 'mae'),
        ('Root Mean Square Error', 'rmse'),
        ('Max Absolute Error', 'max_abs_error'),
        ('Median Absolute Error', 'median_abs_error'),
        ('Mean Relative Error (%)', 'mean_relative_error'),
        ('Median Relative Error (%)', 'median_relative_error'),
    ]
    
    for label, key in metrics:
        val1 = errors1[key]
        val2 = errors2[key]
        
        if 'relative' in key.lower():
            val1_str = f"{val1*100:.4f}%" if not np.isnan(val1) else "N/A"
            val2_str = f"{val2*100:.4f}%" if not np.isnan(val2) else "N/A"
        else:
            val1_str = f"{val1:.6f}" if not np.isnan(val1) else "N/A"
            val2_str = f"{val2:.6f}" if not np.isnan(val2) else "N/A"
        
        print(f"{label:<30} {val1_str:<20} {val2_str:<20}")
        
        if not np.isnan(val1) and not np.isnan(val2):
            if val1 < val2:
                print(f"{'':>30} {'✓ BETTER':<20} {'':<20}")
            elif val2 < val1:
                print(f"{'':>30} {'':<20} {'✓ BETTER':<20}")
    
    print("="*80)


def create_comparison_plots(true_loglkls, surr1_loglkls, surr2_loglkls, 
                           errors1, errors2, run1_name, run2_name, output_dir,
                           kappa_sigma1=None, kappa_sigma2=None,
                           true_loglkls_chain=None, surr1_loglkls_chain=None, surr2_loglkls_chain=None,
                           max_points_per_bin=500, n_bins=100):
    """Create residual plot comparing both models in chi-squared space.
    
    Args:
        kappa_sigma1, kappa_sigma2: kappa_sigma values from configs for legend labels
        true_loglkls_chain, surr1_loglkls_chain, surr2_loglkls_chain: Optional chain sample evaluations for left panel
        max_points_per_bin: Maximum number of points to plot per chi^2 bin (density cap)
        n_bins: Number of bins to use for density-based rejection
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    true_chi2 = -2 * true_loglkls
    surr1_chi2 = -2 * surr1_loglkls
    surr2_chi2 = -2 * surr2_loglkls
    
    finite_mask = np.isfinite(true_chi2) & np.isfinite(surr1_chi2) & np.isfinite(surr2_chi2)
    true_finite = true_chi2[finite_mask]
    surr1_finite = surr1_chi2[finite_mask]
    surr2_finite = surr2_chi2[finite_mask]
    
    print(f"  Applying density-based downsampling (max {max_points_per_bin} points per bin)...")
    chi2_min, chi2_max = np.min(true_finite), np.max(true_finite)
    bin_edges = np.linspace(chi2_min, chi2_max, n_bins + 1)
    bin_indices = np.digitize(true_finite, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    keep_mask = np.zeros(len(true_finite), dtype=bool)
    last_full_bin = -1 
    for bin_idx in range(n_bins):
        bin_mask = (bin_indices == bin_idx)
        n_in_bin = np.sum(bin_mask)
        
        if n_in_bin > max_points_per_bin:
            bin_point_indices = np.where(bin_mask)[0]
            selected_indices = np.random.choice(bin_point_indices, max_points_per_bin, replace=False)
            keep_mask[selected_indices] = True
            last_full_bin = bin_idx
        else:

            keep_mask[bin_mask] = True
    
    if last_full_bin >= 0:
        chi2_upper = bin_edges[last_full_bin + 1]
    else:
        chi2_upper = chi2_max
    
    true_plot = true_finite[keep_mask]
    surr1_plot = surr1_finite[keep_mask]
    surr2_plot = surr2_finite[keep_mask]
    
    print(f"  Downsampled from {len(true_finite)} to {len(true_plot)} points for plotting")
    print(f"  X-axis upper limit set to χ² = {chi2_upper:.2f} (last full bin)")
    
    has_chain_data = (true_loglkls_chain is not None and 
                      surr1_loglkls_chain is not None and 
                      surr2_loglkls_chain is not None)
    
    if has_chain_data:
        fig, (ax_chain, ax_random) = plt.subplots(1, 2, figsize=(width_inches * 2, width_inches * 0.45))
        
        true_chi2_chain = -2 * true_loglkls_chain
        surr1_chi2_chain = -2 * surr1_loglkls_chain
        surr2_chi2_chain = -2 * surr2_loglkls_chain
        
        finite_mask_chain = np.isfinite(true_chi2_chain) & np.isfinite(surr1_chi2_chain) & np.isfinite(surr2_chi2_chain)
        true_finite_chain = true_chi2_chain[finite_mask_chain]
        surr1_finite_chain = surr1_chi2_chain[finite_mask_chain]
        surr2_finite_chain = surr2_chi2_chain[finite_mask_chain]
        
        chi2_upper_chain = np.percentile(true_finite_chain, 99)
        
        print(f"  Applying density-based downsampling to chain data (max {max_points_per_bin} points per bin)...")
        chi2_min_chain = np.min(true_finite_chain)
        bin_edges_chain = np.linspace(chi2_min_chain, chi2_upper_chain, n_bins + 1)
        bin_indices_chain = np.digitize(true_finite_chain, bin_edges_chain) - 1
        bin_indices_chain = np.clip(bin_indices_chain, 0, n_bins - 1)
        
        keep_mask_chain = np.zeros(len(true_finite_chain), dtype=bool)
        for bin_idx in range(n_bins):
            bin_mask = (bin_indices_chain == bin_idx)
            n_in_bin = np.sum(bin_mask)
            
            if n_in_bin > max_points_per_bin:
                bin_point_indices = np.where(bin_mask)[0]
                selected_indices = np.random.choice(bin_point_indices, max_points_per_bin, replace=False)
                keep_mask_chain[selected_indices] = True
            else:
                keep_mask_chain[bin_mask] = True
        
        true_plot_chain = true_finite_chain[keep_mask_chain]
        surr1_plot_chain = surr1_finite_chain[keep_mask_chain]
        surr2_plot_chain = surr2_finite_chain[keep_mask_chain]
        
        print(f"  Downsampled chain data from {len(true_finite_chain)} to {len(true_plot_chain)} points")
        
        residuals1_chain = surr1_plot_chain - true_plot_chain
        residuals2_chain = surr2_plot_chain - true_plot_chain
        
        in_range_chain = true_plot_chain <= chi2_upper_chain
        true_inrange_chain = true_plot_chain[in_range_chain]
        residuals1_inrange_chain = residuals1_chain[in_range_chain]
        residuals2_inrange_chain = residuals2_chain[in_range_chain]
        
        label1 = rf'$\kappa_\sigma = {kappa_sigma1}$' if kappa_sigma1 is not None else run1_name
        label2 = rf'$\kappa_\sigma = {kappa_sigma2}$' if kappa_sigma2 is not None else run2_name
        
        ax_chain.scatter(true_inrange_chain, residuals2_inrange_chain, alpha=0.4, s=12, label=label2, color='C1', rasterized=True)
        ax_chain.scatter(true_inrange_chain, residuals1_inrange_chain, alpha=0.4, s=12, label=label1, color='C0', rasterized=True)
        ax_chain.axhline(y=0, color='black', linestyle='--', lw=1, alpha=0.5, label='Perfect prediction')
        ax_chain.set_xlim(right=chi2_upper_chain)
        ax_chain.set_xlabel(r'$\chi^2_{\mathrm{true}}$')
        ax_chain.set_ylabel(r'$\chi^2_{\mathrm{surr}} - \chi^2_{\mathrm{true}}$')
        ax_chain.legend(loc='best')
        ax_chain.grid(alpha=0.3, linewidth=0.5)
        
        ax = ax_random
    else:
        fig, ax = plt.subplots(1, 1, figsize=(width_inches, width_inches * 0.5))
    
    residuals1 = surr1_plot - true_plot
    residuals2 = surr2_plot - true_plot
    
    in_range = true_plot <= chi2_upper
    true_inrange = true_plot[in_range]
    residuals1_inrange = residuals1[in_range]
    residuals2_inrange = residuals2[in_range]
    
    label1 = rf'$\kappa_\sigma = {kappa_sigma1}$' if kappa_sigma1 is not None else run1_name
    label2 = rf'$\kappa_\sigma = {kappa_sigma2}$' if kappa_sigma2 is not None else run2_name
    
    ax.scatter(true_inrange, residuals1_inrange, alpha=0.4, s=12, label=label1, color='C0', rasterized=True)
    ax.scatter(true_inrange, residuals2_inrange, alpha=0.4, s=12, label=label2, color='C1', rasterized=True)
    ax.axhline(y=0, color='black', linestyle='--', lw=1, alpha=0.5, label='Perfect prediction')
    
    ax.set_xlim(right=chi2_upper)
    ax.set_xlabel(r'$\chi^2_{\mathrm{true}}$')
    if not has_chain_data:
        ax.set_ylabel(r'$\chi^2_{\mathrm{surr}} - \chi^2_{\mathrm{true}}$')
    ax.legend(loc='best')
    ax.grid(alpha=0.3, linewidth=0.5)
    
    if has_chain_data:
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.95, wspace=0.2)
    else:
        fig.subplots_adjust(left=0.12, right=0.975, bottom=0.16, top=0.95)
    
    output_file = output_dir / f'{timestamp}_residual_plot_chi2.pdf'
    plt.savefig(output_file, dpi=150)
    plt.close()
    
    print(f"\nResidual plot saved to: {output_file}")


def save_results(output_dir, true_loglkls, surr1_loglkls, surr2_loglkls, 
                 errors1, errors2, samples, run1_name, run2_name):
    """Save comparison results to file."""
    output_dir = Path(output_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    results = {
        'run1_name': run1_name,
        'run2_name': run2_name,
        'true_loglkls': true_loglkls,
        'surr1_loglkls': surr1_loglkls,
        'surr2_loglkls': surr2_loglkls,
        'errors1': {k: v for k, v in errors1.items() if k not in ['abs_errors', 'relative_errors']},
        'errors2': {k: v for k, v in errors2.items() if k not in ['abs_errors', 'relative_errors']},
        'samples': samples,
        'timestamp': timestamp
    }
    
    results_file = output_dir / f'{timestamp}_comparison_results.pkl'
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to: {results_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare banana likelihood models from two runs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python compare_banana_models.py \\
      results/20251216_173527_banana \\
      results/20251217_191837_banana_k23 \\
      -n 5000 -it 9
        """
    )
    
    parser.add_argument('run1', help='Path to first run directory')
    parser.add_argument('run2', help='Path to second run directory')
    parser.add_argument('-it', '--iteration', type=int, default=9,
                       help='Iteration to compare (default: 9)')
    parser.add_argument('-n', '--n-samples', type=int, default=2000,
                       help='Number of test samples (default: 2000)')
    parser.add_argument('-s', '--strategy', choices=['lhs', 'random', 'grid'], default='lhs',
                       help='Sampling strategy for test points (default: lhs)')
    parser.add_argument('-o', '--output', default=None,
                       help='Output directory (default: comparison_results)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--parallel', action='store_true',
                       help='Use MPI for parallel evaluation of true likelihood')
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = 'comparison_results'
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    run1_name = Path(args.run1).name
    run2_name = Path(args.run2).name
    
    print("="*80)
    print("BANANA LIKELIHOOD MODEL COMPARISON")
    print("="*80)
    print(f"\nRun 1: {args.run1} ({run1_name})")
    print(f"Run 2: {args.run2} ({run2_name})")
    print(f"Iteration: {args.iteration}")
    print(f"Test samples: {args.n_samples}")
    print(f"Sampling strategy: {args.strategy}")
    print(f"Output directory: {output_dir}")
    print()
    
    print("Loading models...")
    surr1, true_likelihood1, cfg1 = load_model_and_likelihood(args.run1, args.iteration)
    surr2, true_likelihood2, cfg2 = load_model_and_likelihood(args.run2, args.iteration)
    
    print(f"  Model 1: {cfg1.wrapper} - kappa_sigma={getattr(cfg1, 'kappa_sigma', 'N/A')}")
    print(f"  Model 2: {cfg2.wrapper} - kappa_sigma={getattr(cfg2, 'kappa_sigma', 'N/A')}")
    
    true_likelihood = true_likelihood1
    param_names = true_likelihood.varying_param_names
    print(f"  Parameters: {param_names}")
    print()
    
    print(f"Generating {args.n_samples} test points using {args.strategy} strategy...")
    samples = generate_test_points(true_likelihood, args.n_samples, args.strategy, args.seed)
    print(f"  Generated {len(samples)} test points")
    print()
    
    print("\nLoading chain samples from banana_planck...")
    try:
        chain_samples = load_chain_samples('banana_planck', max_samples=args.n_samples, seed=args.seed)
        print(f"  Loaded {len(chain_samples)} chain samples")
    except Exception as e:
        print(f"  Warning: Could not load chain samples: {e}")
        chain_samples = None
    print()
    
    print("Evaluating models on random prior samples...")
    true_loglkls, surr1_loglkls, surr2_loglkls = evaluate_models(
        samples, true_likelihood, surr1, surr2, args.parallel
    )
    print("  Evaluation complete!")
    print()
    
    if chain_samples is not None:
        print("Evaluating models on chain samples...")
        true_loglkls_chain, surr1_loglkls_chain, surr2_loglkls_chain = evaluate_models(
            chain_samples, true_likelihood, surr1, surr2, args.parallel
        )
        print("  Evaluation complete!")
        print()
    else:
        true_loglkls_chain = None
        surr1_loglkls_chain = None
        surr2_loglkls_chain = None
    
    print("Computing error metrics...")
    errors1 = compute_errors(true_loglkls, surr1_loglkls)
    errors2 = compute_errors(true_loglkls, surr2_loglkls)
    
    print_comparison_table(errors1, errors2, run1_name, run2_name)
    
    print("\nCreating comparison plots...")
    kappa_sigma1 = getattr(cfg1, 'kappa_sigma', None)
    kappa_sigma2 = getattr(cfg2, 'kappa_sigma', None)
    create_comparison_plots(
        true_loglkls, surr1_loglkls, surr2_loglkls,
        errors1, errors2, run1_name, run2_name, output_dir,
        kappa_sigma1=kappa_sigma1, kappa_sigma2=kappa_sigma2,
        true_loglkls_chain=true_loglkls_chain,
        surr1_loglkls_chain=surr1_loglkls_chain,
        surr2_loglkls_chain=surr2_loglkls_chain
    )
    
    save_results(
        output_dir, true_loglkls, surr1_loglkls, surr2_loglkls,
        errors1, errors2, samples, run1_name, run2_name
    )
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
