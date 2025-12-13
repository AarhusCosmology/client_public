# config/run_manager.py

from datetime import datetime
from pathlib import Path
import re

def _strip_yaml_comments(content):
    lines = []
    for line in content.split('\n'):
        cleaned_line = re.sub(r'#.*$', '', line).rstrip()
        if cleaned_line or (lines and not lines[-1]):
            lines.append(cleaned_line)
    while lines and not lines[-1]:
        lines.pop()
    return '\n'.join(lines)

def _save_config_copy(cfg, config_name, config_contents, append=False):
    config_path = Path(config_name)
    
    if append:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{config_path.stem}{config_path.suffix}"
    else:
        filename = config_path.name
    
    clean_contents = _strip_yaml_comments(config_contents)
    
    save_path = Path(cfg.run_dir) / filename
    save_path.write_text(clean_contents)
    
    return filename

def _resolve_config_path(config_name):
    config_path = Path(config_name)
    if not config_path.exists():
        config_path = Path('config') / config_name
    return config_path

def _format_log_entry(cfg, saved_config_name):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return (
        f"{'=' * 80}\n"
        f"{timestamp}\n"
        f"Ran CLiENT in mode \"{cfg.run_mode}\" starting from iteration {cfg.start_it}\n"
        f"Configuration file: {saved_config_name}\n"
    )

def write_run_log(cfg, config_name, append=False):
    log_path = Path(cfg.run_dir) / 'run.log'
    config_path = _resolve_config_path(config_name)
    
    if not append and config_path.exists():
        config_contents = config_path.read_text()
        saved_config_name = _save_config_copy(cfg, config_path.name, config_contents, append=False)
    else:
        yaml_in_dir = list(Path(cfg.run_dir).glob('*.yaml'))
        yaml_in_dir = [f for f in yaml_in_dir if not f.name.startswith('2025')]
        saved_config_name = yaml_in_dir[0].name if yaml_in_dir else config_name
    
    log_entry = _format_log_entry(cfg, saved_config_name)
    
    with log_path.open('a' if append else 'w') as f:
        f.write(log_entry)

def append_convergence_info(run_dir, iteration, converged):
    log_path = Path(run_dir) / 'run.log'
    
    if converged:
        message = f"CLiENT converged after {iteration + 1} iterations"
    else:
        message = f"CLiENT completed {iteration + 1} iterations without convergence"
    
    with log_path.open('a') as f:
        f.write(message)
