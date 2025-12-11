# likelihood/montepython_wrapper.py

import os
import sys
import tempfile
import shutil
from io import StringIO
from typing import Dict, Any, Optional
import numpy as np

from .base import BaseLikelihood


class MontePythonLikelihood(BaseLikelihood):
    
    def __init__(
        self,
        param_file: str,
        conf_file: str,
        montepython_path: str,
        output_folder: str = './mp_output',
        silent: bool = True
    ):
        super().__init__()
        
        self.param_file = os.path.abspath(param_file)
        self.conf_file = os.path.abspath(conf_file)
        self.montepython_path = os.path.abspath(montepython_path)
        self.output_folder = os.path.abspath(output_folder)
        self.silent = silent
        self._temp_dir = tempfile.mkdtemp(prefix='mp_temp_')
        
        self._validate_paths()
        self._initialize_montepython()
        self._setup_parameters()
    
    def _load_class_exceptions(self):
        try:
            from classy import CosmoSevereError, CosmoComputationError
            self._severe_exception = CosmoSevereError
            self._computation_exception = CosmoComputationError
        except ImportError:
            print("Warning: Could not import CLASS exceptions. Using generic exception handling.")
    
    def _validate_paths(self):
        if not os.path.isfile(self.param_file):
            raise FileNotFoundError(f"Parameter file not found: {self.param_file}")
        if not os.path.isfile(self.conf_file):
            raise FileNotFoundError(f"Configuration file not found: {self.conf_file}")
        if not os.path.isdir(self.montepython_path):
            raise NotADirectoryError(f"MontePython path not found: {self.montepython_path}")
        
        self._load_class_exceptions()
    
    def _initialize_montepython(self):
        if self.montepython_path not in sys.path:
            sys.path.insert(0, self.montepython_path)
        
        try:
            from initialise import initialise as mp_initialise
            import sampler
            from io_mp import CosmologicalModuleError
        except ImportError as e:
            raise ImportError(f"Could not import MontePython modules from {self.montepython_path}. Error: {e}")
        
        command = f'run -p {self.param_file} --conf {self.conf_file} -o {self._temp_dir} --chain-number 0'
        cosmo, data, command_line, _ = self._run_with_optional_silence(lambda: mp_initialise(command))
        
        self.mp = {
            'cosmo': cosmo,
            'data': data,
            'command_line': command_line,
            'compute_lkl': sampler.compute_lkl,
            'get_covmat': sampler.get_covariance_matrix,
            'read_args': sampler.read_args_from_bestfit,
            'cosmo_soft_exception': CosmologicalModuleError,
        }
        
        print(f"MontePython initialized successfully.")
        print(f"  Parameter file: {self.param_file}")
        print(f"  Configuration file: {self.conf_file}")
    
    def _run_with_optional_silence(self, func):
        if not self.silent:
            return func()
        
        stdout_backup = sys.stdout
        stderr_backup = sys.stderr
        try:
            sys.stdout = StringIO()
            sys.stderr = StringIO()
            return func()
        finally:
            sys.stdout = stdout_backup
            sys.stderr = stderr_backup
    
    def _setup_parameters(self):
        from io_mp import get_tex_name
        import re
        
        self.mp_to_wrapper_name = {}
        self.wrapper_to_mp_name = {}
        
        self._process_parameters('varying')
        self._process_parameters('derived')
        
        print(f"Loaded {len(self.param['varying'])} varying parameters:")
        for name, info in self.param['varying'].items():
            print(f"  {name}: range={info['range']}, initial={info['initial']:.6f}")
    
    def _process_parameters(self, param_type):
        from io_mp import get_tex_name
        import re
        
        param_names = self.mp['data'].get_mcmc_parameters([param_type])
        for mp_name in param_names:
            param_dict = self.mp['data'].mcmc_parameters[mp_name]
            wrapper_name = mp_name.replace('*', '')
            
            self.mp_to_wrapper_name[mp_name] = wrapper_name
            self.wrapper_to_mp_name[wrapper_name] = mp_name
            
            tex_name = re.sub('[$*&]', '', get_tex_name(mp_name, param_dict['scale']))
            
            if param_type == 'varying':
                self.param['varying'][wrapper_name] = {
                    'range': param_dict['prior'].prior_range,
                    'initial': param_dict['initial'][0],
                    'sigma': param_dict['initial'][3],
                    'scale': param_dict['scale'],
                    'label': tex_name,
                }
            else:
                self.param['derived'][wrapper_name] = {'label': tex_name}
    
    def _loglkl(self, position: Dict[str, float]) -> float:
        for wrapper_name, value in position.items():
            if wrapper_name in self.wrapper_to_mp_name:
                mp_name = self.wrapper_to_mp_name[wrapper_name]
                self.mp['data'].mcmc_parameters[mp_name]['current'] = value
        
        self.mp['data'].update_cosmo_arguments()
        return self._run_with_optional_silence(lambda: self.mp['compute_lkl'](self.mp['cosmo'], self.mp['data']))
    
    def logprior(self, position: Dict[str, float]) -> float:
        return self.log_uniform_prior(position)
    
    def get_parameter_info(self) -> Dict[str, Any]:
        return {
            'varying': self.param['varying'].copy(),
            'fixed': self.param['fixed'].copy(),
            'derived': self.param['derived'].copy(),
        }
    
    def get_covariance_matrix(self, cov_file: Optional[str] = None) -> np.ndarray:
        if cov_file is not None:
            self.mp['command_line'].cov = cov_file
        
        eigval, eigvec, covmat = self._run_with_optional_silence(
            lambda: self.mp['get_covmat'](self.mp['cosmo'], self.mp['data'], self.mp['command_line'])
        )
        return covmat
    
    def get_initial_position(self) -> Dict[str, float]:
        return {
            wrapper_name: self.mp['data'].mcmc_parameters[self.wrapper_to_mp_name[wrapper_name]]['initial'][0]
            for wrapper_name in self.param['varying'].keys()
        }
    
    def load_bestfit(self, bestfit_file: str) -> Dict[str, float]:
        self._run_with_optional_silence(lambda: self.mp['read_args'](self.mp['data'], bestfit_file))
        
        position = {}
        for wrapper_name in self.param['varying'].keys():
            mp_name = self.wrapper_to_mp_name[wrapper_name]
            param_dict = self.mp['data'].mcmc_parameters[mp_name]
            position[wrapper_name] = param_dict.get('last_accepted', param_dict['initial'][0])
        
        return position
    
    def get_parameter_scales(self) -> Dict[str, float]:
        return {
            wrapper_name: info['sigma'] * info['scale']
            for wrapper_name, info in self.param['varying'].items()
        }
    
    def __del__(self):
        if hasattr(self, '_temp_dir') and os.path.exists(self._temp_dir):
            try:
                shutil.rmtree(self._temp_dir)
            except Exception:
                pass
