import shutil
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from config.config import Config

_SUBDIRS = (
    "likelihood_input",
    "training_data",
    "trained_models",
    "training_history",
    "convergence_stats",
)


def _iteration_numbers(directory: Path, pattern: str) -> list[int]:
    return [int(path.stem.rsplit("_", 1)[-1]) for path in directory.glob(pattern)]


def _local_likelihood_config(run_dir: Path, config: Config) -> Config:
    input_path = run_dir / "likelihood_input" / Path(config.likelihood.input).name
    return replace(
        config,
        likelihood=replace(config.likelihood, input=str(input_path)),
    )


def _start_iteration(run_dir: Path) -> int:
    latest_model = max(
        _iteration_numbers(run_dir / "trained_models", "model_it_*.keras"),
        default=-1,
    )
    latest_data = max(
        _iteration_numbers(run_dir / "training_data", "data_it_*.csv"),
        default=-1,
    )
    start_iteration = max(latest_model, latest_data)
    if start_iteration < 0:
        raise FileNotFoundError(
            f"No training data or trained models in {run_dir}; cannot continue."
        )

    return start_iteration


@dataclass(frozen=True)
class Run:
    config: Config
    run_id: str
    run_dir: Path
    is_new: bool
    retrain: bool
    start_iteration: int
    requested_iterations: int | None

    @classmethod
    def from_args(cls, args):
        path = Path(args.input_or_dir)

        if path.is_file():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = f"{timestamp}_{args.name}" if args.name else timestamp
            return cls(
                config=Config.from_yaml(path),
                run_id=run_id,
                run_dir=Path(args.output) / run_id,
                is_new=True,
                retrain=args.retrain,
                start_iteration=0,
                requested_iterations=args.iterations,
            )

        yaml_files = list(path.glob("*.yaml"))
        if not yaml_files:
            raise FileNotFoundError(f"No YAML configuration found in {path}")

        return cls(
            config=_local_likelihood_config(path, Config.from_yaml(yaml_files[0])),
            run_id=path.name,
            run_dir=path,
            is_new=False,
            retrain=args.retrain,
            start_iteration=(
                args.start if args.start is not None else _start_iteration(path)
            ),
            requested_iterations=args.iterations,
        )

    def create_directories(self, source_config):
        for name in _SUBDIRS:
            (self.run_dir / name).mkdir(parents=True, exist_ok=True)
        source = Path(source_config)
        shutil.copy(source, self.run_dir / source.name)
        likelihood_input = Path(self.config.likelihood.input)
        shutil.copy(likelihood_input, self.likelihood_input / likelihood_input.name)

    # ---- Directory layout ----
    @property
    def likelihood_input(self):
        return self.run_dir / "likelihood_input"

    @property
    def training_data_dir(self):
        return self.run_dir / "training_data"

    @property
    def trained_models_dir(self):
        return self.run_dir / "trained_models"

    @property
    def training_history_dir(self):
        return self.run_dir / "training_history"

    @property
    def convergence_stats_dir(self):
        return self.run_dir / "convergence_stats"

    # ---- Launch behaviour ----
    @property
    def use_convergence(self):
        return self.requested_iterations is None

    @property
    def n_iterations(self):
        if self.requested_iterations is None:
            return self.config.convergence.max_iterations
        return self.requested_iterations + (
            1 if not self.is_new and not self.retrain else 0
        )

    @property
    def final_iteration(self):
        return self.start_iteration + self.n_iterations - 1

    @property
    def mode(self):
        if self.is_new:
            return "new"
        return "continue (retrain)" if self.retrain else "continue"
