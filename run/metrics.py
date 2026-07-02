import json

from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class TrainingMetrics:
    iteration: int
    best_epoch: int
    best_train_loss: float
    best_val_loss: float
    training_time: float
    
@dataclass
class SamplingMetrics:
    iteration: int
    steps_per_walker: int
    acceptance_rate: float
    sampling_time: float
    
@dataclass 
class ResamplingMetrics:
    iteration: int
    pool_size: int        # IS pool drawn from chain
    n_evaluated: int      # candidates sent for true likelihood eval
    n_added: int          # points with finite log-L added to dataset
    resampling_time: float
    acceptance_rate: float = 0.0
    
    def __post_init__(self):
        if self.n_evaluated > 0:
            self.acceptance_rate = self.n_added / self.n_evaluated

@dataclass
class IterationMetrics:
    iteration: int
    total_iteration_time: float
    training: Optional[TrainingMetrics] = None
    sampling: Optional[SamplingMetrics] = None
    resampling: Optional[ResamplingMetrics] = None

class MetricsTracker:
    def __init__(self, results_dir: str, start_iteration: int = 0, preserve_start_metrics: bool = False):
        self.results_dir = Path(results_dir)
        self.metrics_file = self.results_dir / "metrics.log"
        self.metrics_json = self.results_dir / "metrics.json"
        self.start_iteration = start_iteration
        self.preserve_start_metrics = preserve_start_metrics
        self.training_metrics = []
        self.sampling_metrics = []
        self.resampling_metrics = []
        self.iteration_metrics = []
        self.convergence_metrics = {}
        
        if start_iteration > 0 and self.metrics_json.exists():
            self._load_existing_metrics(start_iteration)

    @staticmethod
    def _upsert_metric(metrics_list, metric):
        for i, existing in enumerate(metrics_list):
            if existing.iteration == metric.iteration:
                metrics_list[i] = metric
                return
        metrics_list.append(metric)
    
    def add_training_metrics(self, iteration: int, best_epoch: int,
                           best_train_loss: float, best_val_loss: float,
                           training_time: float) -> None:
        self._upsert_metric(self.training_metrics, TrainingMetrics(
            iteration, best_epoch, best_train_loss, best_val_loss, training_time
        ))
    
    def add_sampling_metrics(self, iteration: int, steps_per_walker: int,
                           acceptance_rate: float, sampling_time: float) -> None:
        self._upsert_metric(self.sampling_metrics, SamplingMetrics(
            iteration, steps_per_walker, acceptance_rate, sampling_time
        ))
    
    def add_resampling_metrics(self, iteration: int, pool_size: int,
                             n_evaluated: int, n_added: int,
                             resampling_time: float) -> None:
        self._upsert_metric(self.resampling_metrics, ResamplingMetrics(
            iteration, pool_size, n_evaluated, n_added, resampling_time
        ))
    
    def add_iteration_metrics(self, iteration: int, total_iteration_time: float) -> None:
        training = next((m for m in self.training_metrics if m.iteration == iteration), None)
        sampling = next((m for m in self.sampling_metrics if m.iteration == iteration), None)
        resampling = next((m for m in self.resampling_metrics if m.iteration == iteration), None)

        self._upsert_metric(self.iteration_metrics, IterationMetrics(
            iteration, total_iteration_time, training, sampling, resampling
        ))
    
    def add_convergence_metrics(self, iteration: int, metric_value: float, converged: bool, metric_name: str = "metric") -> None:
        self.convergence_metrics[iteration] = {
            'metric_name': metric_name,
            'metric_value': float(metric_value),
            'converged': bool(converged)
        }
    
    def save_all_metrics(self) -> None:
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
        f.write("-" * 54 + "\n")
        f.write(f"{'it':<3} | {'best_ep':<7} | {'best_loss':<10} | {'best_val':<10} | {'time':<6}\n")
        f.write("-" * 54 + "\n")
        
        for m in sorted(self.training_metrics, key=lambda x: x.iteration):
            f.write(f"{m.iteration:<3} | {m.best_epoch:<7} | {m.best_train_loss:<10.6f} | {m.best_val_loss:<10.6f} | {m.training_time/60:<6.2f}\n")
        
        f.write("-" * 54 + "\n")
        avg_best_epoch = sum(m.best_epoch for m in self.training_metrics) / len(self.training_metrics)
        avg_best_loss = sum(m.best_train_loss for m in self.training_metrics) / len(self.training_metrics)
        avg_best_val_loss = sum(m.best_val_loss for m in self.training_metrics) / len(self.training_metrics)
        avg_time = sum(m.training_time for m in self.training_metrics) / len(self.training_metrics)
        f.write(f"{'avg':<3} | {avg_best_epoch:<7.1f} | {avg_best_loss:<10.6f} | {avg_best_val_loss:<10.6f} | {avg_time/60:<6.2f}\n")
        f.write("\n\n")
    
    def _write_sampling_metrics(self, f):
        if not self.sampling_metrics:
            return
        
        f.write("Sampling Metrics:\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'it':<3} | {'steps/w':<7} | {'ar':<6} | {'time':<6}\n")
        f.write("-" * 40 + "\n")
        
        for m in sorted(self.sampling_metrics, key=lambda x: x.iteration):
            f.write(f"{m.iteration:<3} | {m.steps_per_walker:<7} | {m.acceptance_rate:<6.3f} | {m.sampling_time/60:<6.2f}\n")
        
        f.write("-" * 40 + "\n")
        avg_steps = sum(m.steps_per_walker for m in self.sampling_metrics) / len(self.sampling_metrics)
        avg_ar = sum(m.acceptance_rate for m in self.sampling_metrics) / len(self.sampling_metrics)
        avg_time = sum(m.sampling_time for m in self.sampling_metrics) / len(self.sampling_metrics)
        f.write(f"{'avg':<3} | {avg_steps:<7.0f} | {avg_ar:<6.3f} | {avg_time/60:<6.2f}\n")
        f.write("\n\n")
    
    def _write_resampling_metrics(self, f):
        if not self.resampling_metrics:
            return
        
        f.write("Resampling Metrics:\n")
        f.write("-" * 46 + "\n")
        f.write(f"{'it':<3} | {'pool':<6} | {'candidates':<10} | {'accepted':<8} | {'time':<6}\n")
        f.write("-" * 46 + "\n")
        
        for m in sorted(self.resampling_metrics, key=lambda x: x.iteration):
            f.write(f"{m.iteration:<3} | {m.pool_size:<6} | {m.n_evaluated:<10} | {m.n_added:<8} | {m.resampling_time/60:<6.2f}\n")
        
        f.write("-" * 46 + "\n")
        tot_pool = sum(m.pool_size for m in self.resampling_metrics)
        tot_evaluated = sum(m.n_evaluated for m in self.resampling_metrics)
        tot_added = sum(m.n_added for m in self.resampling_metrics)
        tot_time = sum(m.resampling_time for m in self.resampling_metrics)
        f.write(f"{'tot':<3} | {tot_pool:<6} | {tot_evaluated:<10} | {tot_added:<8} | {tot_time/60:<6.2f}\n")
        f.write("\n\n")
    
    def _write_iteration_metrics(self, f):
        if not self.iteration_metrics:
            return
        
        f.write("Per-Iteration Runtime:\n")
        f.write("-" * 54 + "\n")
        f.write(f"{'it':<3} | {'total':<7} | {'training':<9} | {'sampling':<9} | {'resampling':<11}\n")
        f.write("-" * 54 + "\n")
        
        training_by_it = {m.iteration: m.training_time for m in self.training_metrics}
        sampling_by_it = {m.iteration: m.sampling_time for m in self.sampling_metrics}
        resampling_by_it = {m.iteration: m.resampling_time for m in self.resampling_metrics}

        for iter_metrics in sorted(self.iteration_metrics, key=lambda x: x.iteration):
            training_time = training_by_it.get(iter_metrics.iteration, 0.0) / 60
            sampling_time = sampling_by_it.get(iter_metrics.iteration, 0.0) / 60
            resampling_time = resampling_by_it.get(iter_metrics.iteration, 0.0) / 60
            total_time = iter_metrics.total_iteration_time / 60

            f.write(f"{iter_metrics.iteration:<3} | {total_time:<7.2f} | {training_time:<9.2f} | {sampling_time:<9.2f} | {resampling_time:<11.2f}\n")
        
        f.write("-" * 54 + "\n")
        tot_total = sum(iter_metrics.total_iteration_time for iter_metrics in self.iteration_metrics) / 60
        tot_training = sum(training_by_it.get(iter_metrics.iteration, 0.0) for iter_metrics in self.iteration_metrics) / 60
        tot_sampling = sum(sampling_by_it.get(iter_metrics.iteration, 0.0) for iter_metrics in self.iteration_metrics) / 60
        tot_resampling = sum(resampling_by_it.get(iter_metrics.iteration, 0.0) for iter_metrics in self.iteration_metrics) / 60
        f.write(f"{'tot':<3} | {tot_total:<7.2f} | {tot_training:<9.2f} | {tot_sampling:<9.2f} | {tot_resampling:<11.2f}\n")
    
    def _write_convergence_metrics(self, f):
        if not self.convergence_metrics:
            return
        
        # Get metric name from first entry (all should be the same)
        first_metrics = next(iter(self.convergence_metrics.values()))
        metric_name = first_metrics.get('metric_name', 'metric')
        
        f.write(f"Convergence Metrics ({metric_name}):\n")
        f.write("-" * 30 + "\n")
        f.write(f"{'it':<3} | {'value':<12} | {'converged':<9}\n")
        f.write("-" * 30 + "\n")
        
        for iteration in sorted(self.convergence_metrics.keys()):
            metrics = self.convergence_metrics[iteration]
            converged_str = "True" if metrics['converged'] else "False"
            f.write(f"{iteration:<3} | {metrics['metric_value']:<12.8f} | {converged_str:<9}\n")
        f.write("\n\n")
    
    def _save_comprehensive_metrics(self) -> None:
        # metrics.json is the structured source of truth (reloaded on continuation);
        # metrics.log is a derived, write-only human-readable view.
        self._save_json()
        with open(self.metrics_file, 'w') as f:
            self._write_header(f)
            self._write_training_metrics(f)
            self._write_sampling_metrics(f)
            self._write_resampling_metrics(f)
            self._write_convergence_metrics(f)
            self._write_iteration_metrics(f)

    def _save_json(self) -> None:
        data = {
            'training': [asdict(m) for m in sorted(self.training_metrics, key=lambda x: x.iteration)],
            'sampling': [asdict(m) for m in sorted(self.sampling_metrics, key=lambda x: x.iteration)],
            'resampling': [asdict(m) for m in sorted(self.resampling_metrics, key=lambda x: x.iteration)],
            'iteration': [
                {'iteration': m.iteration, 'total_iteration_time': m.total_iteration_time}
                for m in sorted(self.iteration_metrics, key=lambda x: x.iteration)
            ],
            'convergence': {str(it): v for it, v in self.convergence_metrics.items()},
        }
        with open(self.metrics_json, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_existing_metrics(self, start_iteration: int) -> None:
        with open(self.metrics_json, 'r') as f:
            data = json.load(f)

        training_cut = (lambda it: it <= start_iteration) if self.preserve_start_metrics else (lambda it: it < start_iteration)

        self.training_metrics = [
            TrainingMetrics(**m) for m in data.get('training', [])
            if training_cut(m['iteration'])
        ]
        self.sampling_metrics = [
            SamplingMetrics(**m) for m in data.get('sampling', [])
            if m['iteration'] < start_iteration
        ]
        self.resampling_metrics = [
            ResamplingMetrics(**m) for m in data.get('resampling', [])
            if m['iteration'] < start_iteration
        ]
        self.iteration_metrics = [
            IterationMetrics(**m) for m in data.get('iteration', [])
            if m['iteration'] < start_iteration
        ]
        convergence_cut = (
            (lambda it: it <= start_iteration)
            if self.preserve_start_metrics
            else (lambda it: it < start_iteration)
        )
        self.convergence_metrics = {
            int(it): v for it, v in data.get('convergence', {}).items()
            if convergence_cut(int(it))
        }
