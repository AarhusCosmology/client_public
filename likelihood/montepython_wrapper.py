# likelihood/montepython_wrapper.py

import sys
import tempfile
import numpy as np
from .base import BaseLikelihood


class MontePythonLikelihood(BaseLikelihood):
    def __init__(self, input_file):
        print(f"Loading MontePython likelihood from '{input_file}'...")
        import yaml
        with open('config/montepython.yaml') as f:
            mp_config = yaml.safe_load(f)
        conf_file = mp_config['conf']
        montepython_path = mp_config['path']

        if montepython_path not in sys.path:
            sys.path.append(montepython_path)

        self._tmp_output = tempfile.TemporaryDirectory(prefix="mp_")
        command = f'run -p {input_file} --conf {conf_file} -o {self._tmp_output.name} --chain-number 0 --silent'

        from initialise import initialise as mp_initialise
        from sampler import compute_lkl
        cosmo, data, _, _ = mp_initialise(command)

        self.cosmo = cosmo
        self.data = data
        self.compute_lkl = compute_lkl
        self._param_names = self.data.get_mcmc_parameters(['varying'])
        self._scales = [self.data.mcmc_parameters[n]['scale'] for n in self._param_names]
        self._raw_bounds = self._compute_bounds()
        self._effective_bounds = None
        print(f"MontePython: found {len(self._param_names)} parameters: {', '.join(self._param_names)}")

    @property
    def varying_param_names(self):
        return self._param_names

    def get_param_names(self):
        return self._param_names

    def _compute_bounds(self):
        bounds = {}
        for name in self._param_names:
            param = self.data.mcmc_parameters[name]
            lower, upper = param['initial'][1], param['initial'][2]
            bounds[name] = (
                None if lower is None else lower * param['scale'],
                None if upper is None else upper * param['scale'],
            )
        return bounds

    def get_prior_bounds(self):
        return dict(self._effective_bounds if self._effective_bounds is not None else self._raw_bounds)

    def restrict_prior_bounds(self, n_sigma):
        restricted_bounds = {}
        for name, (lower_orig, upper_orig) in self._raw_bounds.items():
            param = self.data.mcmc_parameters[name]
            fid   = param['initial'][0] * param['scale']
            sigma = param['initial'][3] * param['scale']
            new_lower = max(lower_orig, fid - n_sigma * sigma) if lower_orig is not None else fid - n_sigma * sigma
            new_upper = min(upper_orig, fid + n_sigma * sigma) if upper_orig is not None else fid + n_sigma * sigma
            restricted_bounds[name] = (new_lower, new_upper)
        self._effective_bounds = restricted_bounds
        print(f"Prior bounds restricted to \u00b1{n_sigma}\u03c3 around fiducial values")

    def restore_prior_bounds(self):
        self._effective_bounds = None

    def loglkl(self, position):
        for name, scale in zip(self._param_names, self._scales):
            self.data.mcmc_parameters[name]['current'] = position[name] / scale
        self.data.need_cosmo_update = True
        self.data.update_cosmo_arguments()
        return float(self.compute_lkl(self.cosmo, self.data))

    def logprior(self, position):
        bounds = self.get_prior_bounds()
        for name, (lower, upper) in bounds.items():
            if (lower is not None and position[name] < lower) or (upper is not None and position[name] > upper):
                return -np.inf
        return 0.0

    def logpost(self, position):
        return self.loglkl(position) + self.logprior(position)

    def __del__(self):
        if hasattr(self, '_tmp_output'):
            self._tmp_output.cleanup()


from .base import BaseLikelihood


