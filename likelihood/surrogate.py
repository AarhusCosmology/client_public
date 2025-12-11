# likelihood/surrogate.py

import pickle
import numpy as np
import tensorflow as tf
from typing import Dict, Any

from model.activations import CustomTanh, Alsing
from .base import BaseLikelihood

CUSTOM_OBJECTS = {
    'CustomTanh': CustomTanh,
    'Alsing': Alsing,
}

class EmulatedLikelihood(BaseLikelihood):
    
    def __init__(self, trained_models_path, x_scaler_path, y_scaler_path, true_likelihood=None):
        super().__init__()
        
        self.model = self._load_model(trained_models_path)
        self._load_scalers(x_scaler_path, y_scaler_path)
        
        self.true_likelihood = true_likelihood
        if true_likelihood is not None:
            self._copy_parameters_from_true_likelihood()
    
    def _load_model(self, model_path):
        return tf.keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS, compile=False)
    
    def _load_scalers(self, x_scaler_path, y_scaler_path):
        with open(x_scaler_path, 'rb') as f:
            x_scaler = pickle.load(f)
        with open(y_scaler_path, 'rb') as f:
            y_scaler = pickle.load(f)
        
        self.x_mean = tf.constant(x_scaler.mean_, dtype=tf.float32)
        self.x_scale = tf.constant(x_scaler.scale_, dtype=tf.float32)
        self.y_mean = tf.constant(y_scaler.mean_, dtype=tf.float32)
        self.y_scale = tf.constant(y_scaler.scale_, dtype=tf.float32)
    
    def _copy_parameters_from_true_likelihood(self):
        self.param = {
            'varying': self.true_likelihood.param['varying'].copy(),
            'fixed': self.true_likelihood.param['fixed'].copy(),
            'derived': self.true_likelihood.param['derived'].copy()
        }

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32)], jit_compile=True)
    def _predict_graph(self, x):
        x_scaled = (x - self.x_mean) / self.x_scale
        y_scaled = self.model(x_scaled, training=False)
        return y_scaled * self.y_scale + self.y_mean

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x).astype(np.float32)
        return self._predict_graph(tf.constant(x)).numpy().ravel()
    
    def loglkl_array(self, x: np.ndarray) -> np.ndarray:
        return self.predict(x)
    
    def _loglkl(self, position: Dict[str, float]) -> float:
        param_names = list(self.param['varying'].keys())
        x = np.array([[position[name] for name in param_names]], dtype=np.float32)
        return float(self.predict(x)[0])
    
    def logprior(self, position: Dict[str, float]) -> float:
        if self.true_likelihood is not None:
            return self.true_likelihood.logprior(position)
        else:
            return self.log_uniform_prior(position)
    
    def _check_parameter_bounds(self, x, param_names):
        n_samples = x.shape[0]
        logpriors = np.zeros(n_samples)
        
        for i, param_name in enumerate(param_names):
            param_range = self.param['varying'][param_name].get('range', [None, None])
            if param_range[0] is not None:
                logpriors[x[:, i] < param_range[0]] = -np.inf
            if param_range[1] is not None:
                logpriors[x[:, i] > param_range[1]] = -np.inf
        
        return logpriors
    
    def logprior_array(self, x: np.ndarray) -> np.ndarray:
        if self.true_likelihood is None:
            raise NotImplementedError(
                "Batch logprior requires true_likelihood to be provided for prior bounds"
            )
        
        x = np.atleast_2d(x)
        param_names = list(self.param['varying'].keys())
        return self._check_parameter_bounds(x, param_names)
    
    def logpost_array(self, x: np.ndarray) -> np.ndarray:
        logpriors = self.logprior_array(x)
        loglkls = self.predict(x)
        
        logposts = np.where(np.isfinite(logpriors), loglkls + logpriors, -np.inf)
        return logposts
    
    def get_parameter_info(self) -> Dict[str, Any]:
        if self.true_likelihood is not None:
            return self.true_likelihood.get_parameter_info()
        else:
            return {
                'varying': self.param['varying'].copy(),
                'fixed': self.param['fixed'].copy(),
                'derived': self.param['derived'].copy(),
            }
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32)])
    def _grad_loglkl_graph(self, x):
        batch_size = tf.shape(x)[0]
        
        def compute_single_gradient(i):
            x_i = x[i:i+1, :]
            with tf.GradientTape() as single_tape:
                single_tape.watch(x_i)
                y_i = self._predict_graph(x_i)
                y_i = tf.squeeze(y_i)
            grad_i = single_tape.gradient(y_i, x_i)
            return tf.squeeze(grad_i, axis=0)
        
        gradients = tf.map_fn(
            compute_single_gradient, 
            tf.range(batch_size), 
            fn_output_signature=tf.TensorSpec(shape=[None], dtype=tf.float32)
        )
        return gradients

    def grad_loglkl(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x).astype(np.float32)
        return self._grad_loglkl_graph(tf.constant(x)).numpy()

