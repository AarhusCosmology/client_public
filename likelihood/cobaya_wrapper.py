import numpy as np

from .base import BaseLikelihood
from utils.mpi_utils import print_master

class CobayaLikelihood(BaseLikelihood):
    def __init__(self, yaml_file):
        print_master(f"Loading Cobaya likelihood from '{yaml_file}'...")
        from cobaya.yaml import yaml_load_file
        from cobaya.model import get_model
        self.yaml_file = yaml_file
        self.cobaya_info = yaml_load_file(self.yaml_file)
        self.cobaya_model = get_model(self.cobaya_info)
        self._param_names = self._get_varying_params()
        self._raw_bounds = self._compute_bounds()
        self._effective_bounds = None
        print_master(f"Cobaya: found {len(self._param_names)} parameters: {', '.join(self._param_names)}")

    def get_param_names(self):
        return self._param_names

    def _get_varying_params(self):
        return [
            name for name, info in self.cobaya_info.get('params', {}).items()
            if isinstance(info, dict) and 'prior' in info
        ]

    def _compute_bounds(self):
        bounds = {}
        for name in self._param_names:
            prior = self.cobaya_info['params'][name].get('prior', {})
            lower = prior.get('min', None)
            upper = prior.get('max', None)
            bounds[name] = (lower, upper)
        return bounds

    def get_param_labels(self):
        return [
            self.cobaya_info['params'][p].get('latex', p).replace('$', '')
            for p in self._param_names
        ]

    def get_prior_bounds(self):
        return dict(self._effective_bounds if self._effective_bounds is not None else self._raw_bounds)

    def restrict_prior_bounds(self, n_sigma):
        restricted_bounds = {}
        for name, (lower_orig, upper_orig) in self._raw_bounds.items():
            param_info = self.cobaya_info['params'][name]
            ref = param_info.get('ref', None)
            fid = ref.get('loc', None) if isinstance(ref, dict) else ref
            sigma = ref.get('scale', None) if isinstance(ref, dict) else param_info.get('proposal', None)
            new_lower = max(lower_orig, fid - n_sigma * sigma) if lower_orig is not None else fid - n_sigma * sigma
            new_upper = min(upper_orig, fid + n_sigma * sigma) if upper_orig is not None else fid + n_sigma * sigma
            restricted_bounds[name] = (new_lower, new_upper)
        self._effective_bounds = restricted_bounds
        print_master(f"Prior bounds restricted to \u00b1{n_sigma}\u03c3 around fiducial values")

    def loglkl(self, x):
        position = {name: float(x[j]) for j, name in enumerate(self._param_names)}
        result = self.cobaya_model.logposterior(position, return_derived=False)
        return float(result.logpost)

    def logprior(self, x):
        bounds = self.get_prior_bounds()
        for j, name in enumerate(self._param_names):
            lower, upper = bounds[name]
            if (lower is not None and x[j] < lower) or (upper is not None and x[j] > upper):
                return -np.inf
        return 0.0

    def logpost(self, x):
        return self.loglkl(x) + self.logprior(x)

    def __del__(self):
        if hasattr(self, 'cobaya_model'):
            try:
                self.cobaya_model.close()
            except Exception:
                pass



