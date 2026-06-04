import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class RunDirectory:
    run_dir: Path
    training_data_dir: Path
    model_dir: Path
    history_dir: Path
    convergence_stats_dir: Path
    config_file: Path

    @classmethod
    def from_new(cls, input_path, name=None):
        input_path = Path(input_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(f'results/{timestamp}_{name}') if name else Path(f'results/{timestamp}')
        td = run_dir / 'training_data'
        md = run_dir / 'trained_models'
        hd = run_dir / 'training_history'
        cd = run_dir / 'convergence_stats'
        td.mkdir(parents=True, exist_ok=True)
        md.mkdir(parents=True, exist_ok=True)
        hd.mkdir(parents=True, exist_ok=True)
        cd.mkdir(parents=True, exist_ok=True)
        shutil.copy(input_path, run_dir / input_path.name)
        return cls(run_dir, td, md, hd, cd, config_file=run_dir / input_path.name)

    @classmethod
    def from_existing(cls, run_dir):
        run_dir = Path(run_dir)
        config_file = next(run_dir.glob('*.yaml'))
        return cls(
            run_dir,
            run_dir / 'training_data',
            run_dir / 'trained_models',
            run_dir / 'training_history',
            run_dir / 'convergence_stats',
            config_file=config_file,
        )
