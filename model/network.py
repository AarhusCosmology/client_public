import tensorflow as tf
from .activations import build_activation, CustomTanh, Alsing

_CUSTOM_OBJECTS = {'CustomTanh': CustomTanh, 'Alsing': Alsing}


def build_model(x_train, n_layers, n_neurons, activation):
    """Build a fully-connected model with a Normalization layer baked in.

    The Normalization layer is adapted to x_train so the model handles raw
    (unscaled) inputs at both training and inference time.
    """
    norm = tf.keras.layers.Normalization()
    norm.adapt(x_train)

    inputs = tf.keras.Input(shape=(x_train.shape[1],))
    x = norm(inputs)
    for _ in range(n_layers):
        x = tf.keras.layers.Dense(n_neurons)(x)
        x = build_activation(activation)(x)
    outputs = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    print(
        f"Built model: input={x_train.shape[1]}D → {n_layers}x{n_neurons} "
        f"({activation}) → 1, {model.count_params():,} trainable parameters"
    )
    return model


def load_model(path):
    return tf.keras.models.load_model(path, custom_objects=_CUSTOM_OBJECTS, compile=False)
