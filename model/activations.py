import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="CLiENT")
class Alsing(tf.keras.layers.Layer):
    def __init__(self, initial_beta=1.0, initial_gamma=0.0, **kwargs):
        super().__init__(**kwargs)
        self.initial_beta = initial_beta
        self.initial_gamma = initial_gamma

    def build(self, input_shape):
        units = int(input_shape[-1])

        self.beta = self.add_weight(
            name="beta",
            shape=(units,),
            initializer=tf.keras.initializers.Constant(self.initial_beta),
            trainable=True,
        )

        self.gamma = self.add_weight(
            name="gamma",
            shape=(units,),
            initializer=tf.keras.initializers.Constant(self.initial_gamma),
            trainable=True,
        )

    def call(self, x):
        return (self.gamma + tf.sigmoid(self.beta * x) * (1.0 - self.gamma)) * x

    def get_config(self):
        config = super().get_config()
        config.update({
            "initial_beta": self.initial_beta,
            "initial_gamma": self.initial_gamma,
        })
        return config

def build_activation(name):
    if name == 'alsing':
        return Alsing()
    return tf.keras.layers.Activation(name)
