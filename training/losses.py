import tensorflow as tf

def _wilson_hilferty(kappa, n):
    # Chi-squared quantile approximation via Wilson-Hilferty transform
    mu = 1.0 - 2.0 / (9.0 * n)
    sigma = tf.sqrt(2.0 / (9.0 * n))
    return n * tf.pow(mu + kappa * sigma, 3.0)

def _create_msre(kappa, n, y_global_max):
    def msre(y_true, y_pred):
        denominator = y_true - y_global_max - 0.5 * _wilson_hilferty(kappa, n)
        relative_error = (y_pred - y_true) / denominator
        return tf.reduce_mean(tf.square(relative_error))
    return msre

def build_loss(name, kappa=None, n=None, y_global_max=None):
    if name == 'msre':
        return _create_msre(kappa, n, y_global_max)
    return name
