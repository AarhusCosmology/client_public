import json
from dataclasses import dataclass
from pathlib import Path

import tensorflow as tf

from .base import ParameterInfo


@dataclass(frozen=True)
class SurrogateMetadata:
    parameters: tuple[ParameterInfo, ...]

    @classmethod
    def from_likelihood(cls, likelihood):
        param_names = likelihood.param_names
        param_labels = likelihood.param_labels
        param_scales = likelihood.param_scales
        bounds = likelihood.prior_bounds

        parameters = tuple(
            ParameterInfo(
                name=name,
                label=label,
                scale=scale,
                lower=bounds[name][0],
                upper=bounds[name][1],
            )
            for name, label, scale in zip(param_names, param_labels, param_scales)
        )

        return cls(parameters=parameters)

    @property
    def param_names(self):
        return [param.name for param in self.parameters]

    @property
    def param_labels(self):
        return [param.label for param in self.parameters]

    @property
    def bounds(self):
        return {param.name: (param.lower, param.upper) for param in self.parameters}

    @property
    def scales(self):
        return [param.scale for param in self.parameters]

    def save(self, path):
        data = {
            "parameters": [
                {
                    "name": param.name,
                    "label": param.label,
                    "scale": param.scale,
                    "lower": param.lower,
                    "upper": param.upper,
                    "center": param.center,
                    "sigma": param.sigma,
                }
                for param in self.parameters
            ]
        }

        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))

        parameters = tuple(
            ParameterInfo(
                name=param["name"],
                label=param["label"],
                scale=param.get("scale", 1.0),
                lower=param.get("lower"),
                upper=param.get("upper"),
                center=param.get("center"),
                sigma=param.get("sigma"),
            )
            for param in data["parameters"]
        )

        return cls(parameters=parameters)


class SurrogateLikelihood:
    def __init__(self, model, metadata):
        self.model = model
        self.metadata = metadata
        self._params = metadata.parameters

        # Pre-compute TensorFlow bound tensors for vectorised prior evaluation.
        # TensorFlow tensors cannot contain None, so unbounded sides are mapped
        # to -inf or +inf.
        lower = []
        upper = []
        for param in self._params:
            lower.append(-float("inf") if param.lower is None else param.lower)
            upper.append(float("inf") if param.upper is None else param.upper)
        self._lower = tf.constant(lower, dtype=tf.float32)
        self._upper = tf.constant(upper, dtype=tf.float32)

    def loglkl(self, positions):
        # Evaluate the surrogate model at the input positions.
        return tf.squeeze(self.model(positions, training=False), axis=1)

    def logprior(self, positions):
        # Return zero inside the surrogate bounds and -inf outside them.
        in_bounds = tf.reduce_all(
            (positions >= self._lower) & (positions <= self._upper), axis=1
        )
        return tf.where(in_bounds, 0.0, -float("inf"))

    def logpost(self, positions):
        return self.loglkl(positions) + self.logprior(positions)

    def logpost_and_gradient(self, positions):
        with tf.GradientTape() as tape:
            tape.watch(positions)
            logpost = self.logpost(positions)

        gradient = tape.gradient(logpost, positions)

        return logpost, gradient
