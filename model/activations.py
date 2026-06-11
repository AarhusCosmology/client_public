import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
class Alsing(tf.keras.layers.Layer):
    """
    Per-feature Alsing activation: a(x) = (gamma + sigmoid(beta * x) * (1 - gamma)) * x
    """
    def build(self, input_shape):
        units = int(input_shape[-1])
        self.beta = self.add_weight(
            name="beta",
            shape=(units,),
             initializer=tf.keras.initializers.Ones(),
            trainable=True,
        )
        self.gamma = self.add_weight(
            name="gamma",
            shape=(units,),
            initializer=tf.keras.initializers.Zeros(),
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