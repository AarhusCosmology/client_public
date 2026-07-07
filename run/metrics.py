import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


SECONDS_PER_MINUTE = 60
MINUTES_PER_HOUR = 60


@dataclass
class TrainingMetrics:
    iteration: int
    epoch: int
    train_loss: float
    val_loss: float
    training_time: float


@dataclass
class SamplingMetrics:
    iteration: int
    steps_per_walker: int
    acceptance_rate: float
    sampling_time: float


@dataclass
class AcquisitionMetrics:
    iteration: int
    n_evaluated: int  # candidates sent for true likelihood eval
    n_added: int  # points with finite log-L added to dataset
    dataset_size: int
    acquisition_time: float
    acceptance_rate: float = 0.0

    def __post_init__(self):
        self.acceptance_rate = (
            self.n_added / self.n_evaluated if self.n_evaluated > 0 else 0.0
        )


@dataclass
class IterationMetrics:
    iteration: int
    total_iteration_time: float


@dataclass
class ConvergenceMetrics:
    iteration: int
    metric_name: str
    metric_value: float
    converged: bool


class MetricsTracker:
    def __init__(
        self,
        results_dir: str,
        start_iteration: int = 0,
        preserve_start_metrics: bool = False,
    ):
        self.results_dir = Path(results_dir)
        self.metrics_report = self.results_dir / "metrics.md"
        self.metrics_json = self.results_dir / "metrics.json"
        self.preserve_start_metrics = preserve_start_metrics
        self.training_metrics: dict[int, TrainingMetrics] = {}
        self.sampling_metrics: dict[int, SamplingMetrics] = {}
        self.acquisition_metrics: dict[int, AcquisitionMetrics] = {}
        self.iteration_metrics: dict[int, IterationMetrics] = {}
        self.convergence_metrics: dict[int, ConvergenceMetrics] = {}

        if self.metrics_json.exists() and (start_iteration > 0 or preserve_start_metrics):
            self._load_existing_metrics(start_iteration)

    @staticmethod
    def _store_metric(metrics, metric) -> None:
        metrics[metric.iteration] = metric

    def add_training_metrics(
        self,
        iteration: int,
        epoch: int,
        train_loss: float,
        val_loss: float,
        training_time: float,
    ) -> None:
        self._store_metric(
            self.training_metrics,
            TrainingMetrics(
                iteration=iteration,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                training_time=training_time,
            ),
        )

    def add_sampling_metrics(
        self,
        iteration: int,
        steps_per_walker: int,
        acceptance_rate: float,
        sampling_time: float,
    ) -> None:
        self._store_metric(
            self.sampling_metrics,
            SamplingMetrics(
                iteration=iteration,
                steps_per_walker=steps_per_walker,
                acceptance_rate=acceptance_rate,
                sampling_time=sampling_time,
            ),
        )

    def add_acquisition_metrics(
        self,
        iteration: int,
        n_evaluated: int,
        n_added: int,
        acquisition_time: float,
        dataset_size: int,
    ) -> None:
        self._store_metric(
            self.acquisition_metrics,
            AcquisitionMetrics(
                iteration=iteration,
                n_evaluated=n_evaluated,
                n_added=n_added,
                dataset_size=dataset_size,
                acquisition_time=acquisition_time,
            ),
        )

    def add_iteration_metrics(self, iteration: int, iteration_time: float) -> None:
        self._store_metric(
            self.iteration_metrics,
            IterationMetrics(
                iteration=iteration,
                total_iteration_time=iteration_time,
            ),
        )

    def add_convergence_metrics(
        self,
        iteration: int,
        metric_value: float,
        converged: bool,
        metric_name: str = "metric",
    ) -> None:
        self._store_metric(
            self.convergence_metrics,
            ConvergenceMetrics(
                iteration=iteration,
                metric_name=metric_name,
                metric_value=float(metric_value),
                converged=bool(converged),
            ),
        )

    def save_all_metrics(self) -> None:
        self._save_comprehensive_metrics()

    @staticmethod
    def _sorted_by_iteration(metrics):
        return [metrics[iteration] for iteration in sorted(metrics)]

    @staticmethod
    def _format_duration(seconds: float) -> str:
        sign = "-" if seconds < 0 else ""
        seconds = abs(seconds)

        if seconds < SECONDS_PER_MINUTE:
            return f"{sign}{seconds:.1f}s"

        rounded_seconds = round(seconds)
        minutes, remaining_seconds = divmod(rounded_seconds, SECONDS_PER_MINUTE)
        if minutes < MINUTES_PER_HOUR:
            return f"{sign}{minutes}m {remaining_seconds:02d}s"

        hours, minutes = divmod(minutes, MINUTES_PER_HOUR)
        return f"{sign}{hours}h {minutes:02d}m"

    @staticmethod
    def _average(metrics, attr: str) -> float:
        return sum(getattr(metric, attr) for metric in metrics) / len(metrics)

    def _metric_dicts(self, metrics):
        return [asdict(metric) for metric in self._sorted_by_iteration(metrics)]

    @staticmethod
    def _load_metric_map(data, key, metric_type, keep):
        metrics = {}
        for raw_metric in data.get(key, []):
            iteration = raw_metric["iteration"]
            if keep(iteration):
                metric = metric_type(**raw_metric)
                metrics[metric.iteration] = metric
        return metrics

    @staticmethod
    def _write_table(f, title, headers, rows):
        if not rows:
            return

        f.write(f"## {title}\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(":---:" for _ in headers) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(value) for value in row) + " |\n")
        f.write("\n")

    def _write_header(self, f, generated_at: datetime):
        f.write("# CLiENT Pipeline Metrics\n\n")
        f.write(f"Generated: {generated_at.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    def _write_training_metrics(self, f):
        metrics = self._sorted_by_iteration(self.training_metrics)
        if not metrics:
            return

        rows = [
            [
                m.iteration,
                m.epoch,
                f"{m.train_loss:.6f}",
                f"{m.val_loss:.6f}",
                self._format_duration(m.training_time),
            ]
            for m in metrics
        ]
        rows.append(
            [
                "avg",
                f"{self._average(metrics, 'epoch'):.1f}",
                f"{self._average(metrics, 'train_loss'):.6f}",
                f"{self._average(metrics, 'val_loss'):.6f}",
                self._format_duration(self._average(metrics, "training_time")),
            ]
        )

        self._write_table(
            f=f,
            title="Training Metrics",
            headers=["it", "epoch", "train loss", "val loss", "time"],
            rows=rows,
        )

    def _write_sampling_metrics(self, f):
        metrics = self._sorted_by_iteration(self.sampling_metrics)
        if not metrics:
            return

        rows = [
            [
                m.iteration,
                m.steps_per_walker,
                f"{m.acceptance_rate:.3f}",
                self._format_duration(m.sampling_time),
            ]
            for m in metrics
        ]
        rows.append(
            [
                "avg",
                f"{self._average(metrics, 'steps_per_walker'):.0f}",
                f"{self._average(metrics, 'acceptance_rate'):.3f}",
                self._format_duration(self._average(metrics, "sampling_time")),
            ]
        )

        self._write_table(
            f=f,
            title="Sampling Metrics",
            headers=["it", "steps/walker", "acceptance", "time"],
            rows=rows,
        )

    def _write_acquisition_metrics(self, f):
        metrics = self._sorted_by_iteration(self.acquisition_metrics)
        if not metrics:
            return

        total_evaluated = sum(m.n_evaluated for m in metrics)
        total_added = sum(m.n_added for m in metrics)

        rows = [
            [
                m.iteration,
                m.n_evaluated,
                m.n_added,
                f"{m.acceptance_rate:.3f}",
                m.dataset_size,
                self._format_duration(m.acquisition_time),
            ]
            for m in metrics
        ]
        rows.append(
            [
                "tot",
                total_evaluated,
                total_added,
                "-",
                "-",
                self._format_duration(sum(m.acquisition_time for m in metrics)),
            ]
        )

        self._write_table(
            f=f,
            title="Acquisition Metrics",
            headers=[
                "it",
                "evaluated",
                "added",
                "added/eval",
                "dataset size",
                "time",
            ],
            rows=rows,
        )

    def _write_iteration_metrics(self, f):
        metrics = self._sorted_by_iteration(self.iteration_metrics)
        if not metrics:
            return

        training_by_it = {
            m.iteration: m.training_time for m in self.training_metrics.values()
        }
        sampling_by_it = {
            m.iteration: m.sampling_time for m in self.sampling_metrics.values()
        }
        acquisition_by_it = {
            m.iteration: m.acquisition_time for m in self.acquisition_metrics.values()
        }

        rows = []
        for m in metrics:
            training_time = training_by_it.get(m.iteration, 0.0)
            sampling_time = sampling_by_it.get(m.iteration, 0.0)
            acquisition_time = acquisition_by_it.get(m.iteration, 0.0)
            other_time = (
                m.total_iteration_time
                - training_time
                - sampling_time
                - acquisition_time
            )
            rows.append(
                [
                    m.iteration,
                    self._format_duration(m.total_iteration_time),
                    self._format_duration(training_time),
                    self._format_duration(sampling_time),
                    self._format_duration(acquisition_time),
                    self._format_duration(other_time),
                ]
            )

        total_time = sum(m.total_iteration_time for m in metrics)
        total_training = sum(training_by_it.get(m.iteration, 0.0) for m in metrics)
        total_sampling = sum(sampling_by_it.get(m.iteration, 0.0) for m in metrics)
        total_acquisition = sum(
            acquisition_by_it.get(m.iteration, 0.0) for m in metrics
        )
        rows.append(
            [
                "tot",
                self._format_duration(total_time),
                self._format_duration(total_training),
                self._format_duration(total_sampling),
                self._format_duration(total_acquisition),
                self._format_duration(
                    total_time - total_training - total_sampling - total_acquisition
                ),
            ]
        )

        self._write_table(
            f=f,
            title="Per-Iteration Runtime",
            headers=["it", "total", "training", "sampling", "acquisition", "other"],
            rows=rows,
        )

    def _write_convergence_metrics(self, f):
        metrics = self._sorted_by_iteration(self.convergence_metrics)
        if not metrics:
            return

        self._write_table(
            f=f,
            title="Convergence Metrics",
            headers=["it", "metric", "value", "converged"],
            rows=[
                [
                    m.iteration,
                    m.metric_name,
                    f"{m.metric_value:.8f}",
                    str(m.converged),
                ]
                for m in metrics
            ],
        )

    def _save_comprehensive_metrics(self) -> None:
        # metrics.json is the structured source of truth (reloaded on continuation);
        # metrics.md is a derived human-readable report.
        self._save_json()
        self._write_report(self.metrics_report, datetime.now())

    def _write_report(self, path: Path, generated_at: datetime) -> None:
        with open(path, "w") as f:
            self._write_header(f, generated_at)
            self._write_training_metrics(f)
            self._write_sampling_metrics(f)
            self._write_acquisition_metrics(f)
            self._write_convergence_metrics(f)
            self._write_iteration_metrics(f)

    def _save_json(self) -> None:
        data = {
            "training": self._metric_dicts(self.training_metrics),
            "sampling": self._metric_dicts(self.sampling_metrics),
            "acquisition": self._metric_dicts(self.acquisition_metrics),
            "iteration": self._metric_dicts(self.iteration_metrics),
            "convergence": self._metric_dicts(self.convergence_metrics),
        }
        with open(self.metrics_json, "w") as f:
            json.dump(data, f, indent=2)

    def _load_existing_metrics(self, start_iteration: int) -> None:
        with open(self.metrics_json, "r") as f:
            data = json.load(f)

        def before_start(iteration):
            return iteration < start_iteration

        def through_start(iteration):
            return (
                iteration <= start_iteration
                if self.preserve_start_metrics
                else before_start(iteration)
            )

        self.training_metrics = self._load_metric_map(
            data, "training", TrainingMetrics, through_start
        )
        self.sampling_metrics = self._load_metric_map(
            data, "sampling", SamplingMetrics, before_start
        )
        self.acquisition_metrics = self._load_metric_map(
            data, "acquisition", AcquisitionMetrics, before_start
        )
        self.iteration_metrics = self._load_metric_map(
            data, "iteration", IterationMetrics, before_start
        )
        self.convergence_metrics = self._load_metric_map(
            data, "convergence", ConvergenceMetrics, through_start
        )
