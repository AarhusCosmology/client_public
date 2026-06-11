import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
class CustomTanh(tf.keras.layers.Layer):
    """Scalar-learnable tanh: a(x) = tanh(alpha * x). Kept for loading legacy models."""
    def __init__(self, initial_alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.initial_alpha = initial_alpha

    def build(self, input_shape):
        self.alpha = self.add_weight(
            name='alpha',
            shape=(1,),
            initializer=tf.keras.initializers.Constant(self.initial_alpha),
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        return tf.math.tanh(self.alpha * inputs)

    def get_config(self):
        config = super().get_config()
        config.update({"initial_alpha": self.initial_alpha})
        return config


@tf.keras.utils.register_keras_serializable()
class Alsing(tf.keras.layers.Layer):
    """
    Per-feature Alsing activation: a(x) = (gamma + sigmoid(beta * x) * (1 - gamma)) * x

    beta and gamma are learned independently for each input feature.
    """
    def build(self, input_shape):
        units = int(input_shape[-1])
        self.beta = self.add_weight(
            name="beta",
            shape=(units,),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0),
            trainable=True,
        )
        self.gamma = self.add_weight(
            name="gamma",
            shape=(units,),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0),
            trainable=True,
        )

    def call(self, x):
        return (self.gamma + tf.sigmoid(self.beta * x) * (1.0 - self.gamma)) * x


def build_activation(name):
    """Return an unbuilt Keras activation layer for the given name."""
    if name == 'alsing':
        return Alsing()
    if name == 'custom_tanh':
        return CustomTanh()
    return tf.keras.layers.Activation(name)