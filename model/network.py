import numpy as np
import tensorflow as tf

from .activations import build_activation

@tf.keras.utils.register_keras_serializable(package="CLiENT")
class TargetDenormalization(tf.keras.layers.Layer):
    """
    Fixed affine layer: y_raw = target_mean + target_std * z.

    The mean and standard deviation are stored as non-trainable weights so
    that they are included in ``model.get_weights()`` / ``set_weights()``
    and are therefore updated correctly when the surrogate is refreshed
    in-place between iterations.
    """
    def __init__(self, mean, std, **kwargs):
        super().__init__(**kwargs)
        self.initial_mean = float(mean)
        self.initial_std = float(std)

    def build(self, input_shape):
        self.target_mean = self.add_weight(
            name="target_mean",
            shape=(),
            initializer=tf.keras.initializers.Constant(self.initial_mean),
            trainable=False,
        )
        self.target_std = self.add_weight(
            name="target_std",
            shape=(),
            initializer=tf.keras.initializers.Constant(self.initial_std),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, inputs):
        return self.target_mean + self.target_std * inputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "mean": self.initial_mean,
            "std": self.initial_std,
        })
        return config

def build_model(inputs, targets, n_layers, n_neurons, activation):
    """
    Build a fully-connected model with input Normalization and output
    TargetDenormalization baked in.

    The Normalization layer is adapted to x_train.  The trainable final
    Dense(1) layer predicts a standardised internal variable z; the
    non-trainable TargetDenormalization layer converts it back to raw
    log-likelihood units so the public model output is always in raw units.

    Parameters
    ----------
    inputs : array-like, shape (N, ndim)
        Training inputs used to adapt the input normalisation layer.
    targets : array-like, shape (N,) or (N, 1)
        Training targets in raw log-likelihood units.  Used only to
        compute the target mean and standard deviation for the output
        denormalisation layer.
    """
    target_array = np.asarray(targets, dtype=np.float64).ravel()
    target_mean = float(np.mean(target_array))
    target_std = float(np.std(target_array))

    norm = tf.keras.layers.Normalization()
    norm.adapt(inputs)

    inputs = tf.keras.Input(shape=(inputs.shape[1],))
    x = norm(inputs)
    for _ in range(n_layers):
        x = tf.keras.layers.Dense(n_neurons)(x)
        x = build_activation(activation)(x)
    z_pred = tf.keras.layers.Dense(1, name="standardized_loglkl")(x)
    outputs = TargetDenormalization(mean=target_mean, std=target_std, name="loglkl_denormalization")(z_pred)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def load_model(path):
    return tf.keras.models.load_model(path, compile=False)
