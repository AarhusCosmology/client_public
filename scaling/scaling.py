# scaling/scaling.py

import os
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler

SCALER_MAP = {
    'minmax': MinMaxScaler,
    'standard': StandardScaler,
}

def _create_scaler(scaler_type):
    if scaler_type not in SCALER_MAP:
        raise ValueError(f"Unknown scaler: {scaler_type}. Available: {list(SCALER_MAP.keys())}")
    return SCALER_MAP[scaler_type]()

def make_scalers(cfg):
    return _create_scaler(cfg.x_scaler_type), _create_scaler(cfg.y_scaler_type)

def _save_scaler(scaler, path):
    with open(path, 'wb') as f:
        pickle.dump(scaler, f)

def save_scalers(cfg, x_scaler, y_scaler, iteration):
    x_path = os.path.join(cfg.scaler_dir, f'x_scaler_it_{iteration}.pkl')
    y_path = os.path.join(cfg.scaler_dir, f'y_scaler_it_{iteration}.pkl')
    
    _save_scaler(x_scaler, x_path)
    _save_scaler(y_scaler, y_path)