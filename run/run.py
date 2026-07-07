import shutil

from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Optional

from config.config import Config

_SUBDIRS = (
    'likelihood_input',
    'training_data',
    'trained_models',
    'training_history',
    'convergence_stats',
)

@dataclass(frozen=True)
class Run:
    config: Config
    run_id: str
    run_dir: Path
    is_new: bool
    retrain: bool
    start_iteration: int
    requested_iterations: Optional[int]  # explicit --iterations (None => use convergence)

    @staticmethod
    def _iteration_numbers(directory: Path, pattern: str) -> list[int]:
        if not directory.is_dir():
            return []

        iterations = []
        for path in directory.glob(pattern):
            try:
                iterations.append(int(path.stem.split('_')[-1]))
            except ValueError:
                continue
        return iterations

    @staticmethod
    def _resolve_continuation_likelihood_input(run_dir: Path, config: Config) -> Config:
        """Prefer the run-local copied likelihood input when continuing.

        Priority:
        1) likelihood_input/<original basename> if present
        2) the sole file in likelihood_input/ if exactly one exists
        3) leave config unchanged
        """
        li_dir = run_dir / 'likelihood_input'
        if not li_dir.is_dir():
            return config

        original_name = Path(config.likelihood.input).name
        named_copy = li_dir / original_name
        if named_copy.is_file():
            return replace(
                config,
                likelihood=replace(config.likelihood, input=str(named_copy)),
            )

        files = [p for p in li_dir.iterdir() if p.is_file()]
        if len(files) == 1:
            return replace(
                config,
                likelihood=replace(config.likelihood, input=str(files[0])),
            )

        return config

    @classmethod
    def from_args(cls, args):
        path = Path(args.input_or_dir)
        is_new = path.is_file()

        if not is_new:
            run_dir = path
            yaml_files = list(run_dir.glob('*.yaml'))
            if not yaml_files:
                raise FileNotFoundError(f"No YAML configuration found in {run_dir}")
            config = Config.from_yaml(yaml_files[0])
            config = cls._resolve_continuation_likelihood_input(run_dir, config)

            if args.start is None:
                model_iterations = cls._iteration_numbers(
                    run_dir / 'trained_models',
                    'model_it_*.keras',
                )
                data_iterations = cls._iteration_numbers(
                    run_dir / 'training_data',
                    'data_it_*.csv',
                )
                if not model_iterations and not data_iterations:
                    raise FileNotFoundError(
                        f"No training data or trained models in {run_dir}; cannot continue."
                    )

                latest_model = max(model_iterations) if model_iterations else -1
                latest_data = max(data_iterations) if data_iterations else -1
                start_iteration = max(latest_model, latest_data)
                if latest_data > latest_model:
                    print(f"Auto-detected next untrained iteration: {start_iteration}")
                else:
                    print(f"Auto-detected latest trained iteration: {start_iteration}")
            else:
                start_iteration = args.start
            run_id = run_dir.name
        else:
            config = Config.from_yaml(path)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            run_id = f'{timestamp}_{args.name}' if args.name else timestamp
            run_dir = Path(args.output) / run_id
            start_iteration = 0

        return cls(
            config=config,
            run_id=run_id,
            run_dir=run_dir,
            is_new=is_new,
            retrain=args.retrain,
            start_iteration=start_iteration,
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
        return self.run_dir / 'likelihood_input'

    @property
    def training_data_dir(self):
        return self.run_dir / 'training_data'

    @property
    def trained_models_dir(self):
        return self.run_dir / 'trained_models'

    @property
    def training_history_dir(self):
        return self.run_dir / 'training_history'

    @property
    def convergence_stats_dir(self):
        return self.run_dir / 'convergence_stats'

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
            return 'new'
        return 'continue (retrain)' if self.retrain else 'continue'
