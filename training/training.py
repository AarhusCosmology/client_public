import os
import time
import pickle
import h5py
import tensorflow as tf
from keras.callbacks import EarlyStopping
from training.loss_functions import get_loss_function

def save_training_data(x, y, path):
    with h5py.File(path, 'w') as f:
        data = f.create_group('data')
        data.create_dataset('x', data=x, dtype='f8')
        data.create_dataset('y', data=y, dtype='f8')

def load_training_data(path):
    with h5py.File(path, 'r') as f:
        return f['data']['x'][...], f['data']['y'][...]

def _get_configured_loss(cfg, y_train, y_scaler, n):
    if cfg.loss_func == 'msre':
        y_max, y_std = float(y_train.max()), float(y_scaler.scale_[0])
        return get_loss_function(cfg.loss_func, y_max, cfg.k_sigma, n, y_std)
    return get_loss_function(cfg.loss_func)

def _create_early_stopping_callback(cfg):
    return EarlyStopping(
        monitor='val_loss',
        patience=cfg.patience,
        restore_best_weights=True,
        verbose=1
    )

def _compile_model(model, cfg, loss_function):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss=loss_function,
        jit_compile=True
    )

def _save_training_artifacts(cfg, model, history, iteration):
    history_path = os.path.join(cfg.training_history_dir, f'history_it_{iteration}.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)
    
    model_path = os.path.join(cfg.trained_models_dir, f'trained_model_it_{iteration}.keras')
    model.save(model_path)

def _extract_training_metrics(history, start_time):
    return {
        'epochs_trained': len(history.history['loss']),
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'training_time': time.time() - start_time
    }

def train_model(cfg, model, x_train, y_train, y_scaler=None, iteration=0, return_metrics=False):
    start_time = time.time() if return_metrics else None
    
    dim = x_train.shape[1]
    loss_function = _get_configured_loss(cfg, y_train, y_scaler, dim)
    _compile_model(model, cfg, loss_function)
    
    history = model.fit(
        x_train,
        y_train,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        validation_split=cfg.val_split,
        verbose=2,
        callbacks=[_create_early_stopping_callback(cfg)]
    )
    
    _save_training_artifacts(cfg, model, history, iteration)
    
    if return_metrics:
        return history, _extract_training_metrics(history, start_time)
    return history