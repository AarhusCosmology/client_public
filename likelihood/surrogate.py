# likelihood/surrogate.py

import numpy as np
import tensorflow as tf
from typing import Dict, Any

from .base import BaseLikelihood


class SurrogateLikelihood(BaseLikelihood):
    """Likelihood backed by a trained Keras model.

    The model is expected to output raw log-likelihood values (no y-scaling).
    Input normalization is handled by a Normalization layer baked into the model
    itself, so no external scalers are required.
    """

    def __init__(self, true_likelihood, model):
        super().__init__()
        self.model = model
        self.true_likelihood = true_likelihood
        self.param = {
            'varying': true_likelihood.param['varying'].copy(),
            'fixed':   true_likelihood.param['fixed'].copy(),
            'derived': true_likelihood.param['derived'].copy(),
        }

    # ------------------------------------------------------------------
    # Batch prediction (used by the sampler hot path)
    # ------------------------------------------------------------------

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x).astype(np.float32)
        return tf.squeeze(self.model(x, training=False), axis=1).numpy()

    def loglkl_array(self, x: np.ndarray) -> np.ndarray:
        return self.predict(x)

    def logprior_array(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        param_names = self.varying_param_names
        logpriors = np.zeros(x.shape[0])
        for i, name in enumerate(param_names):
            bounds = self.param['varying'][name].get('range', [None, None])
            if bounds[0] is not None:
                logpriors[x[:, i] < bounds[0]] = -np.inf
            if bounds[1] is not None:
                logpriors[x[:, i] > bounds[1]] = -np.inf
        return logpriors

    def logpost_array(self, x: np.ndarray) -> np.ndarray:
        logpriors = self.logprior_array(x)
        loglkls = self.predict(x)
        return np.where(np.isfinite(logpriors), loglkls + logpriors, -np.inf)

    # ------------------------------------------------------------------
    # BaseLikelihood abstract method implementations
    # ------------------------------------------------------------------

    def _loglkl(self, position: Dict[str, float]) -> float:
        param_names = self.varying_param_names
        x = np.array([[position[name] for name in param_names]], dtype=np.float32)
        return float(self.predict(x)[0])

    def logprior(self, position: Dict[str, float]) -> float:
        return self.true_likelihood.logprior(position)

    def get_parameter_info(self) -> Dict[str, Any]:
        return self.true_likelihood.get_parameter_info()


