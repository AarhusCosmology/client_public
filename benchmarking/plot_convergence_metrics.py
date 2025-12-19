# benchmarking/plot_convergence_metrics.py

import os
import re
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

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


def parse_benchmark_log(log_file):
    """Parse a benchmark diagnostics log file to extract metrics."""
    metrics = {
        'iteration': None,
        'kl_per_param': {},
        'rms_kl': None,
        'surr_max_loglkl': None,
        'surr_loglkl_at_ref': None,
        'ref_loglkl': None,
        'mean_diffs': {},
        'relative_mean_diffs': {},
        'ci_68_surr': {},
        'ci_68_ref': {},
        'ci_95_surr': {},
        'ci_95_ref': {},
    }
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    match = re.search(r'ITERATION (\d+)', content)
    if match:
        metrics['iteration'] = int(match.group(1))
    
    kl_section = re.search(r'Parameter\s+KL \(nats\)\s+KL \(bits\)\s*\n-+\n(.*?)\n\nSummary:', content, re.DOTALL)
    if kl_section:
        for line in kl_section.group(1).strip().split('\n'):
            parts = line.split()
            if len(parts) >= 3:
                param_name = parts[0]
                kl_nats = float(parts[1])
                metrics['kl_per_param'][param_name] = kl_nats
    
    match = re.search(r'RMS KL divergence:\s+([\d.]+)\s+nats', content)
    if match:
        metrics['rms_kl'] = float(match.group(1))
    match = re.search(r'Surrogate.*?maximum of log\(likelihood\):\s+([\-\d.]+)', content)
    if match:
        metrics['surr_max_loglkl'] = float(match.group(1))
    match = re.search(r'Surrogate log\(likelihood\) at reference.*?best-fit:\s+([-\d.]+)', content)
    if match:
        metrics['surr_loglkl_at_ref'] = float(match.group(1))
    
    match = re.search(r'Reference.*?maximum of log\(likelihood\):\s+([-\d.]+)', content)
    if match:
        metrics['ref_loglkl'] = float(match.group(1))
    
    stats_section = re.search(
        r'=== Posterior Statistics ===\s*\nParameter\s+Surr Mean\s+True Mean\s+Mean Diff\s+Rel \(%\).*?\n-+\n(.*?)\n\n',
        content, re.DOTALL
    )
    if stats_section:
        for line in stats_section.group(1).strip().split('\n'):
            parts = line.split()
            if len(parts) >= 5:
                param_name = parts[0]
                mean_diff = float(parts[3])
                rel_diff = float(parts[4])
                metrics['mean_diffs'][param_name] = mean_diff
                metrics['relative_mean_diffs'][param_name] = rel_diff
    
    ci_section = re.search(
        r'=== 68% / 95% Credible Intervals ===\s*\nParameter\s+Surr 68%\s+True 68%\s+Surr 95%\s+True 95%\s*\n-+\n(.*?)\n\n=== END',
        content, re.DOTALL
    )
    if ci_section:
        for line in ci_section.group(1).strip().split('\n'):
            param_match = re.match(r'(\S+)\s+(.*)', line)
            if not param_match:
                continue
            
            param_name = param_match.group(1)
            interval_str = param_match.group(2)
            
            intervals = re.findall(r'\[[\d.]+,\s*[\d.]+\]|[<>]\s*[\d.]+|N/A', interval_str)
            
            if len(intervals) >= 4:
                metrics['ci_68_surr'][param_name] = parse_ci(intervals[0])
                metrics['ci_68_ref'][param_name] = parse_ci(intervals[1])
                metrics['ci_95_surr'][param_name] = parse_ci(intervals[2])
                metrics['ci_95_ref'][param_name] = parse_ci(intervals[3])
    
    return metrics


def parse_ci(ci_str):
    """Parse credible interval string and return (lower, upper) bounds or None."""
    match = re.search(r'\[([\d.]+),\s*([\d.]+)\]', ci_str)
    if match:
        lower = float(match.group(1))
        upper = float(match.group(2))
        return (lower, upper)
    
    if '>' in ci_str or '<' in ci_str or 'N/A' in ci_str:
        return None
    
    return None


def parse_metrics_log(metrics_file):
    """Parse metrics.log to extract R-1 values."""
    r_minus_one = {}
    
    with open(metrics_file, 'r') as f:
        content = f.read()

    conv_section = re.search(
        r'Convergence Metrics \(Gelman-Rubin R-1\):\s*\n-+\s*\nit\s+\|\s+R-1\s+\|\s+converged\s*\n-+\s*\n(.*?)(?:\n\n|\Z)',
        content, re.DOTALL
    )
    
    if conv_section:
        for line in conv_section.group(1).strip().split('\n'):
            parts = line.split('|')
            if len(parts) >= 2:
                try:
                    iteration = int(parts[0].strip())
                    r1_value = float(parts[1].strip())
                    r_minus_one[iteration] = r1_value
                except (ValueError, IndexError):
                    continue
    
    return r_minus_one


def compute_credible_metric(metrics_list, ci_level='68'):
    """
    Compute the credible metric Δ^CM for each iteration.
    
    Δ^CM_i = (|θ_i^+ - θ_{i,0}^+| + |θ_i^- - θ_{i,0}^-|) / (θ_{0,i}^+ - θ_{0,i}^-)
    
    For one-sided intervals: Δ^CM_i = |θ_i^± - θ_{i,0}^±| / |θ_{0,i}^±|
    
    Returns max_i(Δ^CM) for each iteration.
    """
    max_cm = []
    
    ci_surr_key = f'ci_{ci_level}_surr'
    ci_ref_key = f'ci_{ci_level}_ref'
    
    for metrics in metrics_list:
        ci_surr = metrics[ci_surr_key]
        ci_ref = metrics[ci_ref_key]
        
        cm_values = []
        
        for param in ci_surr:
            if param not in ci_ref:
                continue
            
            surr_width = ci_surr[param]
            ref_width = ci_ref[param]
            
            if surr_width is None or ref_width is None:
                continue

            cm = abs(surr_width - ref_width) / ref_width if ref_width > 0 else np.nan
            
            if not np.isnan(cm):
                cm_values.append(cm)
        
        if cm_values:
            max_cm.append(max(cm_values))
        else:
            max_cm.append(np.nan)
    
    return max_cm


def compute_credible_metric_proper(metrics_list):
    """
    Compute the credible metric Δ^CM for 68% and 95% intervals.
    
    Δ^CM_i = (|θ_i^+ - θ_{i,0}^+| + |θ_i^- - θ_{i,0}^-|) / (θ_{0,i}^+ - θ_{0,i}^-)
    
    Returns max_i(Δ^CM) for each iteration for both 68% and 95%.
    """
    max_cm_68 = []
    max_cm_95 = []
    
    for metrics in metrics_list:
        ci_68_surr = metrics['ci_68_surr']
        ci_68_ref = metrics['ci_68_ref']
        ci_95_surr = metrics['ci_95_surr']
        ci_95_ref = metrics['ci_95_ref']
        
        cm_68_values = []
        cm_95_values = []
        
        for param in ci_68_surr:
            if param in ci_68_ref:
                surr_bounds = ci_68_surr[param]
                ref_bounds = ci_68_ref[param]
                
                if surr_bounds is not None and ref_bounds is not None:
                    surr_lower, surr_upper = surr_bounds
                    ref_lower, ref_upper = ref_bounds
                    
                    ref_width = ref_upper - ref_lower
                    if ref_width > 0:
                        delta_cm = (abs(surr_upper - ref_upper) + abs(surr_lower - ref_lower)) / ref_width
                        cm_68_values.append(delta_cm)
        
        for param in ci_95_surr:
            if param in ci_95_ref:
                surr_bounds = ci_95_surr[param]
                ref_bounds = ci_95_ref[param]
                
                if surr_bounds is not None and ref_bounds is not None:
                    surr_lower, surr_upper = surr_bounds
                    ref_lower, ref_upper = ref_bounds
                    
                    ref_width = ref_upper - ref_lower
                    if ref_width > 0:
                        delta_cm = (abs(surr_upper - ref_upper) + abs(surr_lower - ref_lower)) / ref_width
                        cm_95_values.append(delta_cm)
        
        max_cm_68.append(max(cm_68_values) if cm_68_values else np.nan)
        max_cm_95.append(max(cm_95_values) if cm_95_values else np.nan)
    
    return max_cm_68, max_cm_95


def compute_rms_kl_nongaussian(metrics_list, ngp_indices):
    """Compute RMS KL divergence for non-gaussian parameters."""
    rms_kl_ng = []
    
    for metrics in metrics_list:
        kl_per_param = metrics['kl_per_param']
        if not kl_per_param:
            rms_kl_ng.append(np.nan)
            continue
        
        param_names = list(kl_per_param.keys())
        
        kl_values = []
        for idx in ngp_indices:
            if 0 <= idx < len(param_names):
                param_name = param_names[idx]
                kl_values.append(kl_per_param[param_name])
        
        if kl_values:
            rms = np.sqrt(np.mean(np.array(kl_values)**2))
            rms_kl_ng.append(rms)
        else:
            rms_kl_ng.append(np.nan)
    
    return rms_kl_ng


def main():
    parser = argparse.ArgumentParser(description='Plot convergence metrics from benchmark results')
    parser.add_argument('run_dir', help='Path to run directory')
    parser.add_argument('--ngp', default=None, help='Comma-separated list of non-gaussian parameter indices (0-based), e.g., "6,7"')
    parser.add_argument('-o', '--output', default=None, help='Output file path (default: run_dir/benchmark_figures/convergence_metrics.pdf)')
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    benchmark_dir = run_dir / 'benchmark_results'
    metrics_file = run_dir / 'metrics.log'
    
    if not benchmark_dir.exists():
        raise FileNotFoundError(f"Benchmark results directory not found: {benchmark_dir}")
    
    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics log not found: {metrics_file}")
    
    ngp_indices = []
    if args.ngp:
        ngp_indices = [int(x.strip()) for x in args.ngp.split(',')]
        print(f"Non-gaussian parameter indices: {ngp_indices}")

    log_files = sorted(benchmark_dir.glob('*_diagnostics_it_*.log'))
    if not log_files:
        raise FileNotFoundError(f"No benchmark log files found in {benchmark_dir}")
    
    print(f"Found {len(log_files)} benchmark log files")
    
    all_metrics = []
    for log_file in log_files:
        try:
            metrics = parse_benchmark_log(log_file)
            if metrics['iteration'] is not None:
                all_metrics.append(metrics)
        except Exception as e:
            print(f"Warning: Failed to parse {log_file.name}: {e}")
    
    all_metrics.sort(key=lambda x: x['iteration'])
    
    if not all_metrics:
        raise ValueError("No metrics were successfully parsed from log files")
    
    iterations = [m['iteration'] for m in all_metrics]
    print(f"Parsed metrics for iterations: {iterations}")
    
    r_minus_one_dict = parse_metrics_log(metrics_file)
    
    r_minus_one = [r_minus_one_dict.get(it, np.nan) for it in iterations]
    
    rms_kl = [m['rms_kl'] if m['rms_kl'] is not None else np.nan for m in all_metrics]
    rms_kl_ng = compute_rms_kl_nongaussian(all_metrics, ngp_indices) if ngp_indices else [np.nan] * len(all_metrics)
    surr_max_loglkl = [m['surr_max_loglkl'] if m['surr_max_loglkl'] is not None else np.nan for m in all_metrics]
    surr_loglkl_at_ref = [m['surr_loglkl_at_ref'] if m['surr_loglkl_at_ref'] is not None else np.nan for m in all_metrics]
    ref_loglkl = [m['ref_loglkl'] if m['ref_loglkl'] is not None else np.nan for m in all_metrics]
    delta_cm_68, delta_cm_95 = compute_credible_metric_proper(all_metrics)
    
    height_inches = 5
    fig, axes = plt.subplots(3, 1, figsize=(width_inches, height_inches), sharex=True)
    
    ax = axes[0]
    ax.plot(iterations, rms_kl, 'o-', color='C0', markersize=5, label='RMS KL (all)')
    if ngp_indices:
        ax.plot(iterations, rms_kl_ng, 's-', color='C1', markersize=5, label='RMS KL (non-Gaussian)')
    
    valid_r1 = [(it, r1) for it, r1 in zip(iterations, r_minus_one) if not np.isnan(r1)]
    if valid_r1:
        its, r1s = zip(*valid_r1)
        ax.plot(its, r1s, '^-', color='C2', markersize=5, label=r'$R-1$')
    
    ax.set_ylabel('Metric Value')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_yscale('log')
    ax.legend(loc='best', framealpha=0.9)
    
    ax = axes[1]
    ax.plot(iterations, delta_cm_68, 's-', color='C0', markersize=5, label=r'$\Delta^{\rm CM68}$')
    ax.plot(iterations, delta_cm_95, '^-', color='C1', markersize=5, label=r'$\Delta^{\rm CM95}$')
    ax.set_ylabel(r'max($\Delta^{\rm CM}$)')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_yscale('log')
    ax.legend(loc='best', framealpha=0.9)
    
    ax = axes[2]
    ref_loglkl_best = ref_loglkl[0]
    iterations = np.array(iterations)
    surr_loglkl_at_ref = np.array(surr_loglkl_at_ref)
    
    chi2_diff = -2 * (surr_loglkl_at_ref - ref_loglkl_best)
    
    ax.plot(iterations, chi2_diff, 's-', color='C0', markersize=5, label='Surrogate at Reference')
    ax.set_yscale('symlog', linthresh=1, linscale=1)
    
    linthresh = 1
    ax.axhspan(-linthresh, linthresh, alpha=0.15, color='gray', zorder=0, label='Linear region')
    
    ax.axhline(y=0, color='C2', linestyle='--', label='Perfect agreement')
    
    ax.set_ylabel(r'$\chi^2_{\mathrm{surr}} - \chi^2_{\mathrm{best}}$')
    ax.set_xlabel('Iteration')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(loc='best', framealpha=0.9)
    
    from matplotlib.ticker import FuncFormatter
    def custom_formatter(x, pos):
        if x == 1:
            return '1'
        elif x == -1:
            return '-1'
        elif x == 0:
            return '0'
        elif x == 10:
            return '10'
        elif x == -10:
            return '-10'
        else:
            exp = int(np.log10(abs(x)))
            if x > 0:
                return f'$10^{{{exp}}}$'
            else:
                return f'$-10^{{{exp}}}$'
    ax.yaxis.set_major_formatter(FuncFormatter(custom_formatter))
    
    if len(iterations)>0:
        all_axes_flat = axes.flatten()
        all_axes_flat[-1].set_xticks(iterations)
    
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.10, top=0.98, hspace=0.15)
    
    if args.output:
        output_path = Path(args.output)
    else:
        figure_dir = run_dir / 'benchmark_figures'
        figure_dir.mkdir(exist_ok=True)
        output_path = figure_dir / 'convergence_metrics.pdf'
    
    plt.savefig(output_path, format='pdf', dpi=150)
    print(f"\nPlot saved to: {output_path}")
    
    print("\n=== Summary Statistics ===")
    print(f"Iterations analyzed: {min(iterations)} to {max(iterations)}")
    print(f"\nFinal iteration metrics:")
    print(f"  RMS KL divergence: {rms_kl[-1]:.6f} nats")
    if ngp_indices:
        print(f"  RMS KL divergence (non-gaussian): {rms_kl_ng[-1]:.6f} nats")
    if not np.isnan(r_minus_one[-1]):
        print(f"  R-1: {r_minus_one[-1]:.8f}")
    print(f"  Surrogate log(likelihood) at reference: {surr_loglkl_at_ref[-1]:.4f}")
    chi2_diff_final = -2 * (surr_loglkl_at_ref[-1] - ref_loglkl[0])
    print(f"  chi^2_surr - chi^2_best: {chi2_diff_final:.4f}")
    print(f"  max(Δ^CM) 68%: {delta_cm_68[-1]:.6f}")
    print(f"  max(Δ^CM) 95%: {delta_cm_95[-1]:.6f}")
    print(f"  Surrogate maximum log(likelihood): {surr_max_loglkl[-1]:.4f}")
    print(f"  Surrogate log(likelihood) at reference: {surr_loglkl_at_ref[-1]:.4f}")
    print(f"  max(Δ^CM) 68%: {delta_cm_68[-1]:.6f}")
    print(f"  max(Δ^CM) 95%: {delta_cm_95[-1]:.6f}")


if __name__ == '__main__':
    main()