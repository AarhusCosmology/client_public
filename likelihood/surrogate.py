# likelihood/surrogate.py

import json
from pathlib import Path
from dataclasses import dataclass

import tensorflow as tf


@dataclass(frozen=True)
class SurrogateMetadata:
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

    def loglkl(self, positions):
        return tf.squeeze(self.model(positions, training=False), axis=1)

    def logprior(self, positions):
        in_bounds = tf.reduce_all((positions >= self._lower) & (positions <= self._upper), axis=1)
        return tf.where(in_bounds, 0.0, -float('inf'))

    def logpost(self, positions):
        return self.loglkl(positions) + self.logprior(positions)




