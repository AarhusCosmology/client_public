# model/model.py

import tensorflow as tf
from model.activations import CustomTanh, Alsing

CUSTOM_ACTIVATIONS = {
    'custom_tanh': CustomTanh,
    'alsing': Alsing,
}

def apply_activation(x, activation):
    if activation in CUSTOM_ACTIVATIONS:
        return CUSTOM_ACTIVATIONS[activation]()(x)
    return tf.keras.layers.Activation(activation)(x)

def build_model(cfg, n_params=None):
    n_dims = n_params if n_params is not None else getattr(cfg, 'dim', None)
    if n_dims is None:
        raise ValueError("n_params must be provided or cfg.dim must be set")
    
    inputs = tf.keras.Input(shape=(n_dims,))
    x = inputs
    for _ in range(cfg.n_layers):
        x = tf.keras.layers.Dense(cfg.n_neurons)(x)
        x = apply_activation(x, cfg.act_func)
    outputs = tf.keras.layers.Dense(1)(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)
