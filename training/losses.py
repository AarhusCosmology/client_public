import tensorflow as tf


def _chi2_quantile_wilson_hilferty(sigma_level, chi2_dof):
    # Approximate a chi-squared quantile using the Wilson-Hilferty transform.
    # sigma_level is the equivalent standard-normal sigma level.
    center = 1.0 - 2.0 / (9.0 * chi2_dof)
    width = tf.sqrt(2.0 / (9.0 * chi2_dof))
    return chi2_dof * tf.pow(center + sigma_level * width, 3.0)


def _build_msre(sigma_level, chi2_dof, max_loglkl):
    chi2_quantile = _chi2_quantile_wilson_hilferty(sigma_level, chi2_dof)
    half_chi2_quantile = 0.5 * chi2_quantile

    def msre(true_loglkl, pred_loglkl):
        loglkl_scale = true_loglkl - max_loglkl - half_chi2_quantile
        relative_loglkl_error = (pred_loglkl - true_loglkl) / loglkl_scale
        return tf.reduce_mean(tf.square(relative_loglkl_error))

    return msre


def build_loss(name, sigma_level=None, chi2_dof=None, max_loglkl=None):
    if name == "msre":
        return _build_msre(sigma_level, chi2_dof, max_loglkl)
    return name
