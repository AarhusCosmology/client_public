# config/config_loader.py

import yaml
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime

SUBDIRECTORIES = ['scalers', 'training_data', 'trained_models', 'training_history', 'training_chains', 'convergence_stats']

def create_run_directory(run_name=None, base_results_dir='results'):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{run_name}" if run_name else timestamp
    
    run_dir = Path(base_results_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    subdirs = {name: run_dir / name for name in SUBDIRECTORIES}
    for subdir in subdirs.values():
        subdir.mkdir(parents=True, exist_ok=True)
    
    return run_id, str(run_dir), {k: str(v) for k, v in subdirs.items()}

def create_base_namespace(config):
    likelihood = config['likelihood']
    data = config['data']
    model = config['model']
    training = config['training']
    sampling = config['sampling']
    emcee = sampling['emcee']
    convergence = config.get('convergence', {})
    
    return SimpleNamespace(
        wrapper=str(likelihood['wrapper']),
        param=str(likelihood['param']),
        conf=str(likelihood['conf']),
        path=str(likelihood['path']),
        
        x_scaler_type=str(data['scalers']['parameters']),
        y_scaler_type=str(data['scalers']['targets']),
        
        n_samples=int(data['initial']['n_samples']),
        s_strategy=str(data['initial']['strategy']),
        n_std=float(data['initial']['n_std']),
        
        n_candidates=int(data['iterative']['n_candidates']),
        rs_strategy=str(data['iterative']['strategy']),
        k_NN=int(data['iterative']['k_NN']),
        temperature_training=float(data['iterative']['temperature']),
        update_freq=int(data['iterative']['update_freq']),

        n_layers=int(model['n_layers']),
        n_neurons=int(model['n_neurons']),
        act_func=str(model['activation']),
        
        learning_rate=float(training['learning_rate']),
        loss_func=str(training['loss']),
        k_sigma=float(training['k_sigma']),
        epochs=int(training['n_epochs']),
        batch_size=int(training['batch_size']),
        val_split=float(training['val_split']),
        patience=int(training['patience']),

        temperature=float(sampling['temperature']),
        sampling_method=str(sampling['method']),
        save_chains=bool(sampling.get('save_chains', True)),
        
        n_walkers=int(emcee['n_walkers']),
        burn_in=int(emcee['burn_in']),
        max_steps=int(emcee['max_steps']),
        ess_target=int(emcee['ess_target']),
        chunk_size=int(emcee['chunk_size']),
        delta_tau_tol=float(emcee['delta_tau_tol']),
        ac_thin=int(emcee['ac_thin']),
        
        convergence_threshold=float(convergence.get('r_minus_one_threshold', 0.01)),
        convergence_n_samples=100000,
        max_iterations=int(convergence.get('max_iterations', 20)),
    )

def load_config_cli(args):
    with open(args.input) as f:
        config = yaml.safe_load(f)
    
    namespace = create_base_namespace(config)
    
    continue_dir = args.continue_dir or args.mcmc_continue_dir

    if continue_dir:
        run_mode = 'mcmc_continue' if args.mcmc_continue_dir else 'continue'
        if args.start_it is None:
            raise ValueError(f"--start-it must be specified when using --{run_mode.replace('_', '-')}")
        
        run_dir = Path(continue_dir)
        run_id = run_dir.name
        start_it = args.start_it
        subdirs = {name: str(run_dir / name) for name in SUBDIRECTORIES}
    else:
        run_mode = 'default'
        run_id, run_dir_str, subdirs = create_run_directory(args.name, args.output)
        run_dir = Path(run_dir_str)
        start_it = 0

    namespace.run_id = run_id
    namespace.run_dir = str(run_dir)
    namespace.run_mode = run_mode
    namespace.start_it = start_it
    namespace.n_it = args.n_it
    
    namespace.convergence_enabled = (args.n_it is None)
    
    namespace.scaler_dir = subdirs['scalers']
    namespace.training_data_dir = subdirs['training_data']
    namespace.trained_models_dir = subdirs['trained_models']
    namespace.training_history_dir = subdirs['training_history']
    namespace.chains_dir = subdirs['training_chains']
    namespace.convergence_stats_dir = subdirs['convergence_stats']

    return namespace


def load_config(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return create_base_namespace(config)
