# likelihood/surrogate.py

import tensorflow as tf

from .base import BaseLikelihood


class SurrogateLikelihood(BaseLikelihood):
    """Likelihood backed by a trained Keras model.

    The interface is TensorFlow-native throughout: every method takes a
    ``(n, ndim)`` tensor of positions and returns a ``(n,)`` tensor.  Input
    normalization is baked into the model via a Normalization layer, so no
    external scalers are required.  Callers that need NumPy should convert at
    their own boundary with ``.numpy()``.
    """

    def __init__(self, true_likelihood, model):
        self.model = model
        self.true_likelihood = true_likelihood
        self._param_names = true_likelihood.get_param_names()
        self._bounds = true_likelihood.get_prior_bounds()

        # Pre-compute TF bound tensors for the vectorised prior.
        names = self._param_names
        self._lower = tf.constant([self._bounds[n][0] for n in names], dtype=tf.float32)
        self._upper = tf.constant([self._bounds[n][1] for n in names], dtype=tf.float32)

    @property
    def varying_param_names(self):
        return self._param_names

    def get_param_names(self):
        return self._param_names

    def get_prior_bounds(self):
        return dict(self._bounds)

    # ------------------------------------------------------------------
    # TF-native batch interface.  positions: (n, ndim) -> (n,)
    # ------------------------------------------------------------------

    def loglkl(self, positions):
        """Surrogate log-likelihood for a batch of positions."""
        return tf.squeeze(self.model(positions, training=False), axis=1)

    def logprior(self, positions):
        """Uniform log-prior: 0 inside the bounds, -inf outside."""
        in_bounds = tf.reduce_all((positions >= self._lower) & (positions <= self._upper), axis=1)
        return tf.where(in_bounds, 0.0, -float('inf'))

    def logpost(self, positions):
        """Surrogate log-posterior: loglkl + logprior."""
        return self.loglkl(positions) + self.logprior(positions)


