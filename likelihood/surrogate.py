# likelihood/surrogate.py

import json
from pathlib import Path
from dataclasses import dataclass

import tensorflow as tf


@dataclass(frozen=True)
class SurrogateMetadata:
    """Self-describing parameter-space metadata for a run.

    Carries everything needed to reconstruct a SurrogateLikelihood and to
    interpret stored chains (via ``scales``), so a run can be sampled or
    benchmarked later without re-initialising the (expensive) true likelihood.
    """
    param_names: list
    param_labels: list
    bounds: dict          # name -> (lower, upper)
    scales: list

    @classmethod
    def from_likelihood(cls, likelihood):
        return cls(
            param_names=list(likelihood.get_param_names()),
            param_labels=list(likelihood.get_param_labels()),
            bounds={n: tuple(b) for n, b in likelihood.get_prior_bounds().items()},
            scales=list(likelihood.get_param_scales()),
        )

    def save(self, path):
        Path(path).write_text(json.dumps({
            'param_names': self.param_names,
            'param_labels': self.param_labels,
            'bounds': {n: list(b) for n, b in self.bounds.items()},
            'scales': self.scales,
        }, indent=2))

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        return cls(
            param_names=data['param_names'],
            param_labels=data['param_labels'],
            bounds={n: tuple(b) for n, b in data['bounds'].items()},
            scales=data['scales'],
        )


class SurrogateLikelihood:
    """Likelihood backed by a trained Keras model.

    The interface is TensorFlow-native throughout: every method takes a
    ``(n, ndim)`` tensor of positions and returns a ``(n,)`` tensor.  Input
    normalization is baked into the model via a Normalization layer, so no
    external scalers are required.  Callers that need NumPy should convert at
    their own boundary with ``.numpy()``.
    """

    def __init__(self, model, metadata):
        self.model = model
        self.metadata = metadata
        self._param_names = metadata.param_names
        self._param_labels = metadata.param_labels
        self._bounds = metadata.bounds

        # Pre-compute TF bound tensors for the vectorised prior.
        names = self._param_names
        self._lower = tf.constant([self._bounds[n][0] for n in names], dtype=tf.float32)
        self._upper = tf.constant([self._bounds[n][1] for n in names], dtype=tf.float32)

    def get_param_names(self):
        return self._param_names

    def get_param_labels(self):
        return self._param_labels

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




