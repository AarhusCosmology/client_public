# training/loss_functions.py

import tensorflow as tf

def delta_chi2_from_k(kappa, n):
    kappa = tf.cast(kappa, tf.float32)
    n = tf.cast(n, tf.float32)

    mu = 1.0 - 2.0 / (9.0 * n)
    sigma = tf.sqrt(2.0 / (9.0 * n))
    return n * tf.pow(mu + kappa * sigma, 3.0)

def create_msre_loss(y_global_max, kappa, n, y_std):
    delta_chi2_k = delta_chi2_from_k(kappa, n)
    delta = -y_global_max - 0.5 * (delta_chi2_k / y_std)
    
    def mean_square_relative_error(y_true, y_pred):
        denominator = y_true + delta
        relative_error = (y_pred - y_true) / denominator
        return tf.reduce_mean(tf.square(relative_error))
    
    return mean_square_relative_error


def get_loss_function(loss_name, y_global_max=None, kappa=None, n=None, y_std=None):
    if loss_name == 'msre':
        return create_msre_loss(y_global_max, kappa, n, y_std)
    if loss_name in ['mse', 'mae']:
        return loss_name
    return loss_name

