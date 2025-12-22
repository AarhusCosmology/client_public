# likelihood/cobaya_wrapper.py

import os
import tempfile
import shutil
from typing import Dict, Optional, Any
import numpy as np

from .base import BaseLikelihood


class CobayaLikelihood(BaseLikelihood):
    
    def __init__(
        self,
        yaml_file: str,
        output_folder: str = './cobaya_output',
        debug: bool = False
    ):
        super().__init__()
        
        try:
            from cobaya.model import get_model
            from cobaya.yaml import yaml_load_file
        except ImportError:
            raise ImportError(
                "Cobaya is not installed. Install it with: pip install cobaya"
            )
        
        self.yaml_file = os.path.abspath(yaml_file)
        self._temp_dir = tempfile.mkdtemp(prefix='cobaya_temp_')
        self.output_folder = self._temp_dir
        self.debug = debug
        
        self._validate_paths()
        self._load_exceptions()
        
        self.cobaya_info = yaml_load_file(self.yaml_file)
        self._configure_cobaya_settings()
        
        self.cobaya_model = get_model(self.cobaya_info)
        
        self._setup_parameters()
        
        print(f"Cobaya initialized successfully.")
        print(f"  Configuration file: {self.yaml_file}")
        print(f"  Output folder: {self.output_folder}")
    
    def _load_exceptions(self):
        try:
            from classy import CosmoSevereError, CosmoComputationError
            self._severe_exception = CosmoSevereError
            self._computation_exception = CosmoComputationError
        except ImportError:
            print("Warning: Could not import CLASS exceptions. Using generic exception handling.")
    
    def _validate_paths(self):
        if not os.path.isfile(self.yaml_file):
            raise FileNotFoundError(f"YAML file not found: {self.yaml_file}")
    
    def _configure_cobaya_settings(self):
        if self.debug:
            self.cobaya_info['debug'] = True
        else:
            self.cobaya_info['debug'] = 30
    
    def _setup_parameters(self):
        sampled_params_info = self.cobaya_model.parameterization.sampled_params()
        sampled_params_full = self.cobaya_model.parameterization.sampled_params_info()
        
        for param_name in sampled_params_info:
            param_dict = sampled_params_full.get(param_name, {})
            prior_info = param_dict.get('prior', {})
            ref_info = param_dict.get('ref', None)
            proposal = param_dict.get('proposal', None)
            latex = param_dict.get('latex', param_name)
    
            if isinstance(prior_info, dict):
                if 'min' in prior_info and 'max' in prior_info:
                    lower = prior_info['min']
                    upper = prior_info['max']
                elif 'loc' in prior_info and 'scale' in prior_info:
                    dist_type = prior_info.get('dist', 'uniform')
                    if dist_type == 'uniform':
                        lower = prior_info['loc']
                        upper = prior_info['loc'] + prior_info['scale']
                    else:
                        lower, upper = None, None
                else:
                    lower, upper = None, None
            else:
                lower, upper = None, None
            
            if ref_info is not None:
                if isinstance(ref_info, dict):
                    if 'loc' in ref_info:
                        initial = ref_info['loc']
                    elif 'min' in ref_info and 'max' in ref_info:
                        initial = 0.5 * (ref_info['min'] + ref_info['max'])
                    else:
                        initial = 0.5 * (lower + upper) if (lower is not None and upper is not None) else 0.0
                else:
                    initial = float(ref_info)
            else:
                initial = 0.5 * (lower + upper) if (lower is not None and upper is not None) else 0.0
            
            if isinstance(prior_info, dict) and 'scale' in prior_info:
                sigma = prior_info['scale']
            elif ref_info is not None and isinstance(ref_info, dict) and 'scale' in ref_info:
                sigma = ref_info['scale']
            elif proposal is not None:
                sigma = float(proposal)
            else:
                raise ValueError(
                    f"Parameter {param_name}: Must specify 'prior[scale]', 'ref[scale]', or 'proposal' "
                    f"to determine sigma for restricted prior bounds (prioritized in that order as most "
                    f"representative of standard deviation)."
                )
            
            self.param['varying'][param_name] = {
                'range': [lower, upper],
                'initial': initial,
                'sigma': sigma,
                'scale': 1.0,
                'label': latex,
            }
        
        derived_params = self.cobaya_model.parameterization.derived_params()
        derived_params_info = self.cobaya_model.parameterization.derived_params_info()
        for param_name in derived_params:
            param_info = derived_params_info.get(param_name, {})
            latex = param_info.get('latex', param_name) if isinstance(param_info, dict) else param_name
            self.param['derived'][param_name] = {
                'label': latex
            }
        
        print(f"Loaded {len(self.param['varying'])} varying parameters:")
        for name, info in self.param['varying'].items():
            print(f"  {name}: range={info['range']}, initial={info['initial']:.6f}")
    
    def _loglkl(self, position: Dict[str, float]) -> float:
        result = self.cobaya_model.loglikes(position)
        if not result or result[0] is None:
            return -np.inf
        
        loglikes = result[0]
        
        if isinstance(loglikes, dict):
            total_loglike = np.sum([v for v in loglikes.values() if v is not None])
        elif isinstance(loglikes, (np.ndarray, list)):
            total_loglike = np.sum(loglikes)
        else:
            total_loglike = float(loglikes)
        
        return total_loglike
    
    def logprior(self, position: Dict[str, float]) -> float:
        logprior = self.cobaya_model.logprior(position)
        return logprior
    
    def get_parameter_info(self) -> Dict[str, Any]:
        return {
            'varying': self.param['varying'].copy(),
            'fixed': self.param['fixed'].copy(),
            'derived': self.param['derived'].copy(),
        }
    
    def get_covariance_matrix(self, cov_file: Optional[str] = None) -> np.ndarray:
        if cov_file is not None:
            try:
                covmat = np.loadtxt(cov_file)
                return covmat
            except Exception as e:
                print(f"Warning: Could not load covariance from {cov_file}: {e}")
                print("  Falling back to default covariance.")
        
        try:
            covmat = self.cobaya_model.prior.covmat()
            return covmat
        except Exception:
            n_params = len(self.param['varying'])
            covmat = np.zeros((n_params, n_params))
            for i, param_info in enumerate(self.param['varying'].values()):
                sigma = param_info['sigma']
                covmat[i, i] = sigma ** 2
            return covmat
    
    def get_initial_position(self) -> Dict[str, float]:
        return {
            param_name: info['initial']
            for param_name, info in self.param['varying'].items()
        }
    
    def load_bestfit(self, bestfit_file: str) -> Dict[str, float]:
        try:
            data = np.genfromtxt(bestfit_file, names=True, max_rows=1)
            position = {}
            for param_name in self.param['varying'].keys():
                if param_name in data.dtype.names:
                    position[param_name] = float(data[param_name])
            return position
        except Exception:
            pass
        
        position = {}
        with open(bestfit_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    param_name = parts[0]
                    if param_name in self.param['varying']:
                        try:
                            position[param_name] = float(parts[1])
                        except ValueError:
                            pass
        
        return position
    
    def get_parameter_scales(self) -> Dict[str, float]:
        return {
            param_name: info['sigma']
            for param_name, info in self.param['varying'].items()
        }
    
    def __del__(self):
        if hasattr(self, 'cobaya_model'):
            try:
                self.cobaya_model.close()
            except Exception:
                pass
    
        if hasattr(self, '_temp_dir') and os.path.exists(self._temp_dir):
            try:
                shutil.rmtree(self._temp_dir)
            except Exception:
                pass
