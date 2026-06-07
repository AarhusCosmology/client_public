# config/run.py
#
# A single run: the resolved static Config, where it lives on disk, and how it
# was launched (new vs continuation, start iteration, retrain, iteration count).
# Built once from the CLI arguments and broadcast to MPI workers as one object.

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import Config

_SUBDIRS = (
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
    is_continuation: bool
    retrain: bool
    start_iteration: int
    iterations_override: Optional[int]  # explicit --iterations (None => use convergence)

    @classmethod
    def from_args(cls, args):
        """Resolve CLI arguments into a Run (a pure value object; no I/O).

        ``input_or_dir`` is either a YAML input file (new run) or an existing
        run directory (continuation). For new runs, call ``create_directories``
        afterwards to materialise the run directory on disk.
        """
        path = Path(args.input_or_dir)
        is_continuation = path.is_dir() and (path / 'training_data').exists()

        if is_continuation:
            run_dir = path
            yaml_files = list(run_dir.glob('*.yaml'))
            if not yaml_files:
                raise FileNotFoundError(f"No YAML configuration found in {run_dir}")
            config = Config.from_yaml(yaml_files[0])

            if args.start is None:
                models = (run_dir / 'trained_models').glob('trained_model_it_*.keras')
                iterations = [int(f.stem.split('_')[-1]) for f in models]
                if not iterations:
                    raise FileNotFoundError(
                        f"No trained models in {run_dir / 'trained_models'}; cannot continue."
                    )
                start_iteration = max(iterations)
                print(f"Auto-detected latest iteration: {start_iteration}")
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
            is_continuation=is_continuation,
            retrain=args.retrain,
            start_iteration=start_iteration,
            iterations_override=args.iterations,
        )

    def create_directories(self, source_config):
        """Create the run directory tree and archive the input config verbatim.

        Called once on the master for new runs; continuations reuse the existing
        directory and its already-archived config.
        """
        for name in _SUBDIRS:
            (self.run_dir / name).mkdir(parents=True, exist_ok=True)
        source = Path(source_config)
        shutil.copy(source, self.run_dir / source.name)
        likelihood_input = Path(self.config.likelihood.input)
        shutil.copy(likelihood_input, self.run_dir / likelihood_input.name)

    # ---- Directory layout ----
    @property
    def training_data(self):
        return self.run_dir / 'training_data'

    @property
    def trained_models(self):
        return self.run_dir / 'trained_models'

    @property
    def training_history(self):
        return self.run_dir / 'training_history'

    @property
    def convergence_stats(self):
        return self.run_dir / 'convergence_stats'

    # ---- Launch behaviour ----
    @property
    def use_convergence(self):
        """Stop on the convergence criterion rather than a fixed iteration count."""
        return self.iterations_override is None

    @property
    def reuse_initial_model(self):
        """On a continuation without --retrain, reuse the existing model for the
        first iteration instead of retraining it."""
        return self.is_continuation and not self.retrain

    @property
    def n_iterations(self):
        """Number of loop iterations to run for this invocation."""
        if self.iterations_override is None:
            return self.config.convergence.max_iterations
        # Continuations get one extra so the reused-model iteration isn't counted.
        return self.iterations_override + (1 if self.is_continuation else 0)

    @property
    def last_iteration(self):
        return self.start_iteration + self.n_iterations - 1

    @property
    def mode_label(self):
        if not self.is_continuation:
            return 'new'
        return 'continue (retrain)' if self.retrain else 'continue'
