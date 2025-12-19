# metrics/metrics_tracker.py

from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class TrainingMetrics:
    iteration: int
    epochs_trained: int
    final_train_loss: float
    final_val_loss: float
    training_time: float
    
@dataclass
class SamplingMetrics:
    iteration: int
    steps_to_convergence: int
    acceptance_rate: float
    sampling_time: float
    final_max_tau: Optional[float] = None
    converged: bool = True
    
@dataclass 
class ResamplingMetrics:
    iteration: int
    candidates_processed: int
    rejected_emulated: int
    rejected_true: int
    accepted: int
    resampling_time: float
    n_initial_samples: int = 0
    acceptance_rate: float = 0.0
    
    def __post_init__(self):
        if self.candidates_processed > 0:
            self.acceptance_rate = self.accepted / self.candidates_processed

@dataclass
class IterationMetrics:
    iteration: int
    total_iteration_time: float
    training: Optional[TrainingMetrics] = None
    sampling: Optional[SamplingMetrics] = None
    resampling: Optional[ResamplingMetrics] = None

class MetricsTracker:
    def __init__(self, results_dir: str, start_iteration: int = 0):
        self.results_dir = Path(results_dir)
        self.metrics_file = self.results_dir / "metrics.log"
        self.start_iteration = start_iteration
        self.training_metrics = []
        self.sampling_metrics = []
        self.resampling_metrics = []
        self.iteration_metrics = []
        self.convergence_metrics = {}
        
        if start_iteration > 0 and self.metrics_file.exists():
            self._load_existing_metrics(start_iteration)
    
    def add_training_metrics(self, iteration: int, epochs_trained: int, 
                           final_train_loss: float, final_val_loss: float,
                           training_time: float) -> None:
        self.training_metrics.append(TrainingMetrics(
            iteration, epochs_trained, final_train_loss, final_val_loss, training_time
        ))
    
    def add_sampling_metrics(self, iteration: int, steps_to_convergence: int,
                           acceptance_rate: float, sampling_time: float,
                           final_max_tau: Optional[float] = None,
                           converged: bool = True) -> None:
        self.sampling_metrics.append(SamplingMetrics(
            iteration, steps_to_convergence, acceptance_rate, sampling_time, final_max_tau, converged
        ))
    
    def add_resampling_metrics(self, iteration: int, candidates_processed: int,
                             rejected_emulated: int, rejected_true: int,
                             accepted: int, resampling_time: float,
                             n_initial_samples: int = 0) -> None:
        self.resampling_metrics.append(ResamplingMetrics(
            iteration, candidates_processed, rejected_emulated, rejected_true, accepted, resampling_time, n_initial_samples
        ))
    
    def add_iteration_metrics(self, iteration: int, total_iteration_time: float) -> None:
        training = next((m for m in self.training_metrics if m.iteration == iteration), None)
        sampling = next((m for m in self.sampling_metrics if m.iteration == iteration), None)
        resampling = next((m for m in self.resampling_metrics if m.iteration == iteration), None)
        
        self.iteration_metrics.append(IterationMetrics(
            iteration, total_iteration_time, training, sampling, resampling
        ))
    
    def add_convergence_metrics(self, iteration: int, r_minus_one: float, converged: bool) -> None:
        self.convergence_metrics[iteration] = {
            'r_minus_one': float(r_minus_one),
            'converged': bool(converged)
        }
    
    def save_all_metrics(self, suffix: str = None) -> None:
        self._save_comprehensive_metrics()
    
    def save_progress_metrics(self, iteration: int) -> None:
        self._save_comprehensive_metrics()
        print(f"   Progress metrics updated through iteration {iteration}")
    
    def _write_header(self, f):
        f.write("CLiENT Pipeline Metrics\n")
        f.write("=" * 80 + "\n")
        f.write(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n\n")
    
    def _write_training_metrics(self, f):
        if not self.training_metrics:
            return
        
        f.write("Training Metrics:\n")
        f.write("-" * 51 + "\n")
        f.write(f"{'it':<3} | {'epochs':<7} | {'loss':<10} | {'val_loss':<10} | {'time':<6}\n")
        f.write("-" * 51 + "\n")
        
        for m in self.training_metrics:
            f.write(f"{m.iteration:<3} | {m.epochs_trained:<7} | {m.final_train_loss:<10.6f} | {m.final_val_loss:<10.6f} | {m.training_time/60:<6.2f}\n")
        
        f.write("-" * 51 + "\n")
        avg_epochs = sum(m.epochs_trained for m in self.training_metrics) / len(self.training_metrics)
        avg_loss = sum(m.final_train_loss for m in self.training_metrics) / len(self.training_metrics)
        avg_val_loss = sum(m.final_val_loss for m in self.training_metrics) / len(self.training_metrics)
        avg_time = sum(m.training_time for m in self.training_metrics) / len(self.training_metrics)
        f.write(f"{'avg':<3} | {avg_epochs:<7.1f} | {avg_loss:<10.6f} | {avg_val_loss:<10.6f} | {avg_time/60:<6.2f}\n")
        f.write("\n\n")
    
    def _write_sampling_metrics(self, f):
        if not self.sampling_metrics:
            return
        
        f.write("Sampling Metrics:\n")
        f.write("-" * 58 + "\n")
        f.write(f"{'it':<3} | {'steps':<7} | {'ar':<6} | {'max(τ)':<9} | {'converged':<9} | {'time':<6}\n")
        f.write("-" * 58 + "\n")
        
        for m in self.sampling_metrics:
            tau_str = f"{m.final_max_tau:.2f}" if m.final_max_tau is not None else "N/A"
            converged_str = "True" if m.converged else "False"
            f.write(f"{m.iteration:<3} | {m.steps_to_convergence:<7} | {m.acceptance_rate:<6.3f} | {tau_str:<9} | {converged_str:<9} | {m.sampling_time/60:<6.2f}\n")
        
        f.write("-" * 58 + "\n")
        avg_steps = sum(m.steps_to_convergence for m in self.sampling_metrics) / len(self.sampling_metrics)
        avg_ar = sum(m.acceptance_rate for m in self.sampling_metrics) / len(self.sampling_metrics)
        avg_tau = sum(m.final_max_tau for m in self.sampling_metrics if m.final_max_tau is not None) / len([m for m in self.sampling_metrics if m.final_max_tau is not None]) if any(m.final_max_tau is not None for m in self.sampling_metrics) else 0
        avg_time = sum(m.sampling_time for m in self.sampling_metrics) / len(self.sampling_metrics)
        f.write(f"{'avg':<3} | {avg_steps:<7.0f} | {avg_ar:<6.3f} | {avg_tau:<9.2f} | {'-':<9} | {avg_time/60:<6.2f}\n")
        f.write("\n\n")
    
    def _write_resampling_metrics(self, f):
        if not self.resampling_metrics:
            return
        
        f.write("Resampling Metrics:\n")
        f.write("-" * 95 + "\n")
        f.write(f"{'it':<3} | {'processed':<9} | {'accepted':<8} | {'ar':<7} | {'rejected (surrogate)':<20} | {'rejected':<8} | {'evals':<6} | {'time':<6}\n")
        f.write("-" * 95 + "\n")
        
        for m in self.resampling_metrics:
            initial_samples = m.n_initial_samples if m.iteration == 0 else 0
            evals = initial_samples + m.accepted + m.rejected_true
            f.write(f"{m.iteration:<3} | {m.candidates_processed:<9} | {m.accepted:<8} | {m.acceptance_rate:<7.4f} | {m.rejected_emulated:<20} | {m.rejected_true:<8} | {evals:<6} | {m.resampling_time/60:<6.2f}\n")
        
        f.write("-" * 95 + "\n")
        tot_processed = sum(m.candidates_processed for m in self.resampling_metrics)
        tot_accepted = sum(m.accepted for m in self.resampling_metrics)
        tot_ar = tot_accepted / tot_processed if tot_processed > 0 else 0
        tot_rej_surrogate = sum(m.rejected_emulated for m in self.resampling_metrics)
        tot_rej_true = sum(m.rejected_true for m in self.resampling_metrics)
        initial_samples_total = sum(m.n_initial_samples if m.iteration == 0 else 0 for m in self.resampling_metrics)
        tot_evals = initial_samples_total + tot_accepted + tot_rej_true
        tot_time = sum(m.resampling_time for m in self.resampling_metrics)
        f.write(f"{'tot':<3} | {tot_processed:<9} | {tot_accepted:<8} | {tot_ar:<7.4f} | {tot_rej_surrogate:<20} | {tot_rej_true:<8} | {tot_evals:<6} | {tot_time/60:<6.2f}\n")
        f.write("\n\n")
    
    def _write_iteration_metrics(self, f):
        if not self.iteration_metrics:
            return
        
        f.write("Per-Iteration Runtime:\n")
        f.write("-" * 54 + "\n")
        f.write(f"{'it':<3} | {'total':<7} | {'training':<9} | {'sampling':<9} | {'resampling':<11}\n")
        f.write("-" * 54 + "\n")
        
        for iter_metrics in self.iteration_metrics:
            training_time = iter_metrics.training.training_time/60 if iter_metrics.training else 0
            sampling_time = iter_metrics.sampling.sampling_time/60 if iter_metrics.sampling else 0  
            resampling_time = iter_metrics.resampling.resampling_time/60 if iter_metrics.resampling else 0
            total_time = iter_metrics.total_iteration_time/60
            
            f.write(f"{iter_metrics.iteration:<3} | {total_time:<7.2f} | {training_time:<9.2f} | {sampling_time:<9.2f} | {resampling_time:<11.2f}\n")
        
        f.write("-" * 54 + "\n")
        tot_total = sum(iter_metrics.total_iteration_time for iter_metrics in self.iteration_metrics) / 60
        tot_training = sum(iter_metrics.training.training_time if iter_metrics.training else 0 for iter_metrics in self.iteration_metrics) / 60
        tot_sampling = sum(iter_metrics.sampling.sampling_time if iter_metrics.sampling else 0 for iter_metrics in self.iteration_metrics) / 60
        tot_resampling = sum(iter_metrics.resampling.resampling_time if iter_metrics.resampling else 0 for iter_metrics in self.iteration_metrics) / 60
        f.write(f"{'tot':<3} | {tot_total:<7.2f} | {tot_training:<9.2f} | {tot_sampling:<9.2f} | {tot_resampling:<11.2f}\n")
    
    def _write_convergence_metrics(self, f):
        if not self.convergence_metrics:
            return
        
        f.write("Convergence Metrics (Gelman-Rubin R-1):\n")
        f.write("-" * 33 + "\n")
        f.write(f"{'it':<3} | {'R-1':<12} | {'converged':<9}\n")
        f.write("-" * 33 + "\n")
        
        for iteration in sorted(self.convergence_metrics.keys()):
            metrics = self.convergence_metrics[iteration]
            converged_str = "True" if metrics['converged'] else "False"
            f.write(f"{iteration:<3} | {metrics['r_minus_one']:<12.8f} | {converged_str:<9}\n")
        f.write("\n\n")
    
    def _save_comprehensive_metrics(self) -> None:
        with open(self.metrics_file, 'w') as f:
            self._write_header(f)
            self._write_training_metrics(f)
            self._write_sampling_metrics(f)
            self._write_resampling_metrics(f)
            self._write_convergence_metrics(f)
            self._write_iteration_metrics(f)
    
    def _load_existing_metrics(self, start_iteration: int) -> None:
        import re
        
        with open(self.metrics_file, 'r') as f:
            content = f.read()
        
        def parse_training_section(content):
            match = re.search(r'Training Metrics:\n-+\nit.*?\n-+\n(.*?)\n-+', content, re.DOTALL)
            if not match:
                return []
            metrics = []
            for line in match.group(1).strip().split('\n'):
                parts = [p.strip() for p in line.split('|')]
                if parts[0] == 'avg':
                    continue
                it, epochs, loss, val_loss, time = int(parts[0]), int(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])*60
                if it <= start_iteration:
                    metrics.append(TrainingMetrics(it, epochs, loss, val_loss, time))
            return metrics
        
        def parse_sampling_section(content):
            match = re.search(r'Sampling Metrics:\n-+\nit.*?\n-+\n(.*?)\n-+', content, re.DOTALL)
            if not match:
                return []
            metrics = []
            for line in match.group(1).strip().split('\n'):
                parts = [p.strip() for p in line.split('|')]
                if parts[0] == 'avg':
                    continue
                it, steps, ar = int(parts[0]), int(parts[1]), float(parts[2])
                tau_str = parts[3]
                tau = float(tau_str) if tau_str != 'N/A' else None
                converged = parts[4] == 'True'
                time = float(parts[5])*60
                if it < start_iteration:
                    metrics.append(SamplingMetrics(it, steps, ar, time, tau, converged))
            return metrics
        
        def parse_resampling_section(content):
            match = re.search(r'Resampling Metrics:\n-+\nit.*?\n-+\n(.*?)\n-+', content, re.DOTALL)
            if not match:
                return []
            metrics = []
            for line in match.group(1).strip().split('\n'):
                parts = [p.strip() for p in line.split('|')]
                if parts[0] == 'tot':
                    continue
                it, processed, accepted = int(parts[0]), int(parts[1]), int(parts[2])
                rejected_emulated, rejected_true = int(parts[4]), int(parts[5])
                evals, time = int(parts[6]), float(parts[7])*60
                n_initial = evals - accepted - rejected_true if it == 0 else 0
                if it < start_iteration:
                    metrics.append(ResamplingMetrics(it, processed, rejected_emulated, rejected_true, accepted, time, n_initial))
            return metrics
        
        def parse_convergence_section(content):
            match = re.search(r'Convergence Metrics.*?\n-+\nit.*?\n-+\n(.*?)(?:\n\n|$)', content, re.DOTALL)
            if not match:
                return {}
            metrics = {}
            for line in match.group(1).strip().split('\n'):
                parts = [p.strip() for p in line.split('|')]
                it = int(parts[0])
                if it < start_iteration:
                    metrics[it] = {'r_minus_one': float(parts[1]), 'converged': parts[2] == 'True'}
            return metrics
        
        def parse_iteration_section(content):
            match = re.search(r'Per-Iteration Runtime:\n-+\nit.*?\n-+\n(.*?)\n-+', content, re.DOTALL)
            if not match:
                return []
            metrics = []
            for line in match.group(1).strip().split('\n'):
                parts = [p.strip() for p in line.split('|')]
                if parts[0] == 'tot':
                    continue
                it, total = int(parts[0]), float(parts[1])*60
                if it < start_iteration:
                    metrics.append(IterationMetrics(it, total))
            return metrics
        
        self.training_metrics = parse_training_section(content)
        self.sampling_metrics = parse_sampling_section(content)
        self.resampling_metrics = parse_resampling_section(content)
        self.convergence_metrics = parse_convergence_section(content)
        self.iteration_metrics = parse_iteration_section(content)