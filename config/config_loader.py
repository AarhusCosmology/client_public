# config/config_loader.py

import yaml
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime

SUBDIRECTORIES = ['training_data', 'trained_models', 'training_history', 'training_chains', 'convergence_stats']

def _find_latest_iteration(run_dir):
    trained_models_dir = run_dir / 'trained_models'
    iterations = []
    for f in trained_models_dir.glob('trained_model_it_*.keras'):
        iterations.append(int(f.stem.split('_')[-1]))

    if not iterations:
        raise FileNotFoundError(
            f"No trained model files found in {trained_models_dir}. "
            f"Cannot continue from this run directory."
        )

    return max(iterations)

def _find_yaml_in_dir(directory):
    yaml_files = list(Path(directory).glob('*.yaml'))
    return yaml_files[0]

def create_run_directory(run_name=None, base_results_dir='results'):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{run_name}" if run_name else timestamp

    run_dir = Path(base_results_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    subdirs = {name: run_dir / name for name in SUBDIRECTORIES}
    for subdir in subdirs.values():
        subdir.mkdir(parents=True, exist_ok=True)

    return run_id, run_dir, subdirs

def create_base_namespace(config):
    likelihood = config['likelihood']
    data       = config['data']
    model      = config['model']
    training   = config['training']
    sampling   = config['sampling']
    convergence = config['convergence']

    return SimpleNamespace(
        # likelihood
        wrapper=str(likelihood['wrapper']),
        param=str(likelihood['input']),

        # data / initial
        n_samples=int(data['n_samples']),
        n_sigma=float(data['n_sigma']),

        # data / augmentation
        n_augment=int(data['n_augment']),
        n_neighbors=int(data['n_neighbors']),
        target_temperature=float(data['target_temperature']),
        pool_factor=int(data.get('pool_factor', 20)),

        # model
        n_layers=int(model['n_layers']),
        n_neurons=int(model['n_neurons']),
        act_func=str(model['activation']),

        # training
        learning_rate=float(training['learning_rate']),
        loss_func=str(training['loss']),
        kappa_sigma=float(training['kappa_sigma']),
        epochs=int(training['n_epochs']),
        batch_size=int(training['batch_size']),
        validation_split=float(training['validation_split']),
        patience=int(training['patience']),

        # sampling (flat)
        sampler=str(sampling['sampler']),
        temperature=float(sampling['temperature']),
        n_walkers=int(sampling['n_walkers']),
        burn_in=int(sampling['burn_in']),
        max_steps=int(sampling['max_steps']),

        # convergence
        convergence_metric=str(convergence.get('metric', 'marginal_r_minus_one')),
        convergence_threshold=float(convergence['threshold']),
        max_iterations=int(convergence.get('max_iterations', 20)),
    )

def load_config_cli(args):
    if args.mode == 'continue':
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        yaml_file = _find_yaml_in_dir(run_dir)
        with open(yaml_file) as f:
            config = yaml.safe_load(f)

        namespace = create_base_namespace(config)

        run_mode = 'retrain_continue' if args.retrain else 'skip_retrain_continue'
        run_id = run_dir.name

        if args.start is None:
            start_it = _find_latest_iteration(run_dir)
            print(f"Auto-detected latest iteration: {start_it}")
        else:
            start_it = args.start

        subdirs = {name: run_dir / name for name in SUBDIRECTORIES}
    else:
        with open(args.input) as f:
            config = yaml.safe_load(f)

        namespace = create_base_namespace(config)

        run_mode = 'default'
        run_id, run_dir, subdirs = create_run_directory(args.name, args.output)
        start_it = 0

    namespace.run_id    = run_id
    namespace.run_dir   = run_dir
    namespace.run_mode  = run_mode
    namespace.start_it  = start_it
    namespace.n_it      = args.iterations
    namespace.retrain   = getattr(args, 'retrain', False)

    namespace.convergence_enabled = (args.iterations is None)

    namespace.training_data_dir   = subdirs['training_data']
    namespace.trained_models_dir  = subdirs['trained_models']
    namespace.training_history_dir = subdirs['training_history']
    namespace.chains_dir          = subdirs['training_chains']
    namespace.convergence_stats_dir = subdirs['convergence_stats']

    return namespace


def load_config(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return create_base_namespace(config)



