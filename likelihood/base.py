# likelihood/base.py

from abc import ABC, abstractmethod
from typing import Dict, List, Any
import numpy as np

class BaseLikelihood(ABC):
    class NullException(Exception):
        pass
    
    def __init__(self):
        self.param = {'varying': {}, 'fixed': {}, 'derived': {}}
        self._original_ranges = {}
        self._computation_exception = self.NullException
        self._severe_exception = Exception
    
    @property
    def varying_param_names(self) -> List[str]:
        return list(self.param['varying'].keys())
    
    @property
    def fixed_param_names(self) -> List[str]:
        return list(self.param['fixed'].keys())
    
    @property
    def derived_param_names(self) -> List[str]:
        return list(self.param['derived'].keys())
    
    @abstractmethod
    def _loglkl(self, position: Dict[str, float]) -> float:
        pass
    
    def _build_full_position(self, position: Dict[str, float]) -> Dict[str, float]:
        full_position = {**position}
        full_position.update({name: params['fixed_value'] for name, params in self.param['fixed'].items()})
        return full_position
    
    def loglkl(self, position: Dict[str, float]) -> float:
        full_position = self._build_full_position(position)
        
        try:
            return self._loglkl(full_position)
        except self._computation_exception as e:
            print(f"Computation exception occurred: {e}. Returning -inf.")
            return -np.inf
        except self._severe_exception:
            raise
    
    @abstractmethod
    def logprior(self, position: Dict[str, float]) -> float:
        pass
    
    def logpost(self, position: Dict[str, float]) -> float:
        lp = self.logprior(position)
        if not np.isfinite(lp):
            return -np.inf
        return self.loglkl(position) + lp
    
    def outside_of_prior_bound(self, position: Dict[str, float]) -> bool:
        for param_name, value in position.items():
            if param_name not in self.param['varying']:
                continue
            
            lower, upper = self.param['varying'][param_name].get('range', [None, None])
            if (lower is not None and value < lower) or (upper is not None and value > upper):
                return True
        return False
    
    def log_uniform_prior(self, position: Dict[str, float]) -> float:
        if self.outside_of_prior_bound(position):
            return -np.inf
        return 0.0
    
    def set_fixed_parameters(self, fixed_param_dict: Dict[str, float]):
        for param_name, fixed_value in fixed_param_dict.items():
            if param_name in self.param['varying']:
                self.param['fixed'][param_name] = self.param['varying'][param_name].copy()
                self.param['fixed'][param_name]['fixed_value'] = fixed_value
                del self.param['varying'][param_name]
    
    def _store_original_ranges(self):
        if not self._original_ranges:
            for param_name, param_info in self.param['varying'].items():
                self._original_ranges[param_name] = param_info.get('range', [None, None]).copy()
    
    def _calculate_restricted_bounds(self, initial, sigma, n_std, original_range):
        lower = initial - n_std * sigma
        upper = initial + n_std * sigma
        
        if original_range[0] is not None:
            lower = max(lower, original_range[0])
        if original_range[1] is not None:
            upper = min(upper, original_range[1])
        
        return [lower, upper]
    
    def restrict_prior(self, n_std: float = None):
        self._store_original_ranges()
        
        if n_std is None:
            return
        
        for param_name, param_info in self.param['varying'].items():
            original_range = self._original_ranges[param_name]
            sigma = param_info.get('sigma')
            initial = param_info.get('initial')
            
            if sigma is not None and initial is not None:
                param_info['range'] = self._calculate_restricted_bounds(initial, sigma, n_std, original_range)
            elif original_range[0] is None or original_range[1] is None:
                raise ValueError(
                    f"Parameter {param_name} has infinite prior bounds and no sigma/initial "
                    f"specified. Cannot restrict prior."
                )
    
    def restore_prior(self):
        if not self._original_ranges:
            return
        
        for param_name in self.param['varying'].keys():
            if param_name in self._original_ranges:
                self.param['varying'][param_name]['range'] = self._original_ranges[param_name].copy()
    
    def get_prior_bounds(self) -> Dict[str, List[float]]:
        bounds = {}
        for param_name, param_info in self.param['varying'].items():
            lower, upper = param_info.get('range', [None, None])
            
            if lower is None or upper is None:
                raise ValueError(f"Parameter {param_name} has infinite prior bounds. Use restrict_prior() to set finite bounds for sampling.")
            
            bounds[param_name] = [lower, upper]
        
        return bounds
    
    @abstractmethod
    def get_parameter_info(self) -> Dict[str, Any]:
        pass
