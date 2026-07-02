import os
import sys
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import tensorflow as tf
from scipy import stats


class TeeOutput:
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.log_file = None
        self.original_stdout = sys.stdout
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    def __enter__(self):
        self.log_file = open(self.log_file_path, "w", encoding="utf-8")
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


@dataclass(frozen=True)
class DiagnosticsConfig:
    iteration: int
    config_yaml: object
    run_dir: object
    thin: int
    n_steps: int
    chains_path: object = None
    surrogate_sampler: str = "ensemble"
    reference_sampler: str = None
    surrogate_convergence_available: bool = True


def compute_kl_divergence_kde(samples_p, samples_q, param_indices=None, max_samples_kde=50000, rng=None):
    samples_p = _as_2d(samples_p)
    samples_q = _as_2d(samples_q)

    if param_indices is not None:
        samples_p = samples_p[:, param_indices]
        samples_q = samples_q[:, param_indices]

    rng = np.random.default_rng() if rng is None else rng
    samples_p_kde = _subsample(samples_p, max_samples_kde, rng)
    samples_q_kde = _subsample(samples_q, max_samples_kde, rng)

    per_param_kl = {}
    for i in range(samples_p.shape[1]):
        kde_p = stats.gaussian_kde(samples_p_kde[:, i])
        kde_q = stats.gaussian_kde(samples_q_kde[:, i])
        x_min = min(samples_p[:, i].min(), samples_q[:, i].min())
        x_max = max(samples_p[:, i].max(), samples_q[:, i].max())
        x_range = x_max - x_min
        x_grid = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, 1000)
        p_vals = np.maximum(kde_p(x_grid), 1e-10)
        q_vals = np.maximum(kde_q(x_grid), 1e-10)
        mask = p_vals > 1e-8
        integrand = np.where(mask, p_vals * np.log(p_vals / q_vals), 0)
        per_param_kl[i] = max(0.0, np.trapezoid(integrand, x_grid))

    return sum(per_param_kl.values()), per_param_kl


def print_diagnostics(
    samples,
    reference_samples,
    param_names,
    getdist_names,
    config,
    surrogate=None,
):
    print(f"=== BENCHMARK DIAGNOSTICS - ITERATION {config.iteration} ===")
    print(f"Configuration: {config.config_yaml}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Run directory: {config.run_dir}")
    print(f"Surrogate sampler: {config.surrogate_sampler}")
    if config.reference_sampler:
        print(f"Reference sampler: {config.reference_sampler}")
    print(f"Chain shape: {samples.samples.shape}")
    print(f"Thin factor: {config.thin}")
    print(f"MCMC steps: {config.n_steps}")
    if config.chains_path:
        print(f"Reference chains loaded from: {config.chains_path}")
    print("=" * 70)

    _print_convergence(
        samples,
        param_names,
        f"Surrogate - {config.surrogate_sampler}",
        config.surrogate_sampler,
        config.surrogate_convergence_available,
    )

    if reference_samples:
        ref_label = (
            f"Reference ({config.reference_sampler})"
            if config.reference_sampler
            else "Reference/True Chains"
        )
        _print_convergence(reference_samples, param_names, ref_label)

    _print_posterior_statistics(samples, reference_samples, param_names)
    if reference_samples:
        _print_kl_divergence(samples, reference_samples, param_names)
    _print_bestfit(samples, reference_samples, param_names, config, surrogate)
    _print_credible_intervals(samples, reference_samples, param_names, getdist_names)

    print("\n=== END DIAGNOSTICS ===")


def _as_2d(samples):
    samples = np.asarray(samples)
    return samples.reshape(-1, 1) if samples.ndim == 1 else samples


def _subsample(samples, max_samples, rng):
    if len(samples) <= max_samples:
        return samples
    return samples[rng.choice(len(samples), max_samples, replace=False)]


def _safe_relative_percent(diff, reference):
    if reference == 0:
        return np.nan
    return diff / abs(reference) * 100


def _format_percent(value):
    return "N/A" if np.isnan(value) else f"{value:.1f}"


def _print_convergence(
    samples,
    param_names,
    label,
    sampler_name=None,
    convergence_available=True,
):
    print(f"\n=== Convergence Diagnostics ({label}) ===")
    if not convergence_available:
        print("N/A: fewer than two surrogate chain blocks are available.")
        return

    try:
        gelman_rubin = samples.getGelmanRubin()
        print(f"Gelman-Rubin statistic: {gelman_rubin:.4f}")
        if gelman_rubin > 1.1:
            print("  WARNING: Gelman-Rubin > 1.1, chain may not be converged!")
        if str(sampler_name).lower() == "aies":
            print("\nWARNING: Gelman-Rubin diagnostic is not reliable with the AIES")
            print("         ensemble sampler as walkers are not independent chains.")
    except Exception as e:
        print(f"Gelman-Rubin statistic: N/A ({e})")

    print("\nEffective sample sizes:")
    for i, param_name in enumerate(param_names):
        try:
            ess = samples.getEffectiveSamples(i)
            print(f"  {param_name}: {ess:.0f}")
        except Exception as e:
            print(f"  {param_name}: N/A ({e})")


def _print_posterior_statistics(samples, reference_samples, param_names):
    print("\n=== Posterior Statistics ===")
    surrogate_means = samples.getMeans()
    surrogate_stds = np.sqrt(samples.getVars())

    if not reference_samples:
        print(f"{'Parameter':<20} {'Mean':>12} {'Std':>10}")
        print("-" * 45)
        for i, param_name in enumerate(param_names):
            print(f"{param_name:<20} {surrogate_means[i]:>12.4f} {surrogate_stds[i]:>10.4f}")
        return

    reference_means = reference_samples.getMeans()
    reference_stds = np.sqrt(reference_samples.getVars())

    header = (
        f"{'Parameter':<20} {'Surr Mean':>12} {'True Mean':>12} "
        f"{'Mean Diff':>10} {'Rel (%)':>8} {'Surr Std':>10} "
        f"{'True Std':>10} {'Std Diff':>10} {'Rel (%)':>8}"
    )
    print(header)
    print("-" * len(header))

    for i, param_name in enumerate(param_names):
        mean_diff = abs(surrogate_means[i] - reference_means[i])
        std_diff = abs(surrogate_stds[i] - reference_stds[i])
        relative_mean_diff = _safe_relative_percent(mean_diff, reference_means[i])
        relative_std_diff = _safe_relative_percent(std_diff, reference_stds[i])
        print(
            f"{param_name:<20} "
            f"{surrogate_means[i]:>12.4f} {reference_means[i]:>12.4f} "
            f"{mean_diff:>10.4f} {_format_percent(relative_mean_diff):>8} "
            f"{surrogate_stds[i]:>10.4f} {reference_stds[i]:>10.4f} "
            f"{std_diff:>10.4f} {_format_percent(relative_std_diff):>8}"
        )


def _print_kl_divergence(samples, reference_samples, param_names):
    print("\n=== KL Divergence Analysis ===")
    print("Computing D_KL(True || Surrogate) for marginal distributions...")
    print("(measures information lost when using surrogate instead of true posterior)")

    try:
        print(f"\n{'Parameter':<20} {'KL (nats)':>15} {'KL (bits)':>15}")
        print("-" * 52)

        kl_values = []
        for i, param_name in enumerate(param_names):
            kl_nats = compute_kl_divergence_kde(
                reference_samples.samples[:, i:i + 1],
                samples.samples[:, i:i + 1],
                param_indices=[0],
                max_samples_kde=5000,
            )[1][0]
            kl_bits = kl_nats / np.log(2)
            kl_values.append(kl_nats)
            print(f"{param_name:<20} {kl_nats:>15.6f} {kl_bits:>15.6f}")

        rms_kl = np.sqrt(np.mean(np.array(kl_values) ** 2))

        print("\nSummary:")
        print(f"  RMS KL divergence: {rms_kl:.6f} nats ({rms_kl / np.log(2):.6f} bits)")
        print("\nInterpretation:")
        print("  < 0.01 nats: Excellent agreement")
        print("  0.01-0.1 nats: Good agreement")
        print("  0.1-0.5 nats: Moderate discrepancy")
        print("  > 0.5 nats: Significant discrepancy")

    except Exception as e:
        print(f"Error computing KL divergence: {e}")


def _print_bestfit(samples, reference_samples, param_names, config, surrogate):
    print("\n=== Maximum Log-Likelihood Samples (from MCMC chains) ===")
    surrogate_bestfit = samples.samples[np.argmin(samples.loglikes)]
    print(
        f"Surrogate ({config.surrogate_sampler}) maximum of log(likelihood): "
        f"{-min(samples.loglikes):.4f}"
    )

    if not reference_samples:
        print(f"{'Parameter':<20} {'Surrogate MAP':>15}")
        print("-" * 37)
        for i, param_name in enumerate(param_names):
            print(f"{param_name:<20} {surrogate_bestfit[i]:>15.4f}")
        return

    reference_bestfit = reference_samples.samples[np.argmin(reference_samples.loglikes)]
    print(
        f"Reference ({config.reference_sampler}) maximum of log(likelihood): "
        f"{-min(reference_samples.loglikes):.4f}"
    )
    if surrogate is not None:
        reference_bestfit_tensor = tf.cast(reference_bestfit.reshape(1, -1), tf.float32)
        surr_at_true_map = float(surrogate.logpost(reference_bestfit_tensor).numpy()[0])
        print(
            f"Surrogate log(likelihood) at reference "
            f"({config.reference_sampler}) best-fit: {surr_at_true_map:.4f}"
        )

    print()
    header = f"{'Parameter':<20} {'Surr MAP':>12} {'True MAP':>12} {'Diff':>10} {'Rel (%)':>8}"
    print(header)
    print("-" * len(header))

    map_diffs = []
    for i, param_name in enumerate(param_names):
        diff = abs(surrogate_bestfit[i] - reference_bestfit[i])
        rel_diff = _safe_relative_percent(diff, reference_bestfit[i])
        map_diffs.append(diff)
        print(
            f"{param_name:<20} "
            f"{surrogate_bestfit[i]:>12.4f} {reference_bestfit[i]:>12.4f} "
            f"{diff:>10.4f} {_format_percent(rel_diff):>8}"
        )

    print(f"\nMAP difference RMS: {np.sqrt(np.mean(np.array(map_diffs) ** 2)):.4f}")


def _print_credible_intervals(samples, reference_samples, param_names, getdist_names):
    print("\n=== 68% / 95% Credible Intervals ===")
    surrogate_stats = samples.getMargeStats()

    if reference_samples:
        reference_stats = reference_samples.getMargeStats()
        print(
            f"{'Parameter':<20} {'Surr 68%':<22} {'True 68%':<22} "
            f"{'Surr 95%':<22} {'True 95%':<22}"
        )
        print("-" * 110)

        for param_name, sample_name in zip(param_names, getdist_names):
            surrogate_param = surrogate_stats.parWithName(sample_name)
            reference_param = reference_stats.parWithName(sample_name)
            s68, s95 = _limits(surrogate_param)
            t68, t95 = _limits(reference_param)
            print(
                f"{param_name:<20} {_format_limit(s68):<22} {_format_limit(t68):<22} "
                f"{_format_limit(s95):<22} {_format_limit(t95):<22}"
            )
        return

    print(f"{'Parameter':<20} {'68% Interval':<25} {'95% Interval':<25}")
    print("-" * 72)

    for param_name, sample_name in zip(param_names, getdist_names):
        surrogate_param = surrogate_stats.parWithName(sample_name)
        s68, s95 = _limits(surrogate_param)
        print(f"{param_name:<20} {_format_limit(s68):<25} {_format_limit(s95):<25}")


def _limits(param_stats):
    if param_stats is None or not param_stats.limits:
        return None, None
    limit_68 = param_stats.limits[0]
    limit_95 = param_stats.limits[1] if len(param_stats.limits) > 1 else None
    return limit_68, limit_95


def _format_limit(limit):
    if limit is None:
        return "N/A"
    if getattr(limit, "twotail", False):
        return f"[{limit.lower:.4f}, {limit.upper:.4f}]"
    if getattr(limit, "onetail_lower", 0):
        return f"> {limit.lower:.4f}"
    if getattr(limit, "onetail_upper", 0):
        return f"< {limit.upper:.4f}"
    return "N/A"
