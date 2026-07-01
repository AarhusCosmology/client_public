from .base import BaseLikelihood, ParameterInfo

class CobayaLikelihood(BaseLikelihood):
    def __init__(self, yaml_path):
        # Load the Cobaya modules
        from cobaya.yaml import yaml_load_file
        from cobaya.model import get_model

        # Store the Cobaya objects as attributes
        self.cobaya_info = yaml_load_file(yaml_path)
        self.cobaya_model = get_model(self.cobaya_info)

        # Let BaseLikelihood extract parameters and initialize bounds state
        super().__init__()

    def _extract_params(self):
        params = []
        for name, info in self.cobaya_info['params'].items():
            # Keep only entries with metadata and a prior;
            # these define the coordinates varied by this wrapper.
            if not isinstance(info, dict):
                continue
            prior = info.get('prior')
            if prior is None:
                continue

            # Use Cobaya's reference point as the center for optional
            # n-sigma prior restriction. Fall back to the proposal width
            # when ref does not provide an explicit scale.
            ref = info.get('ref')
            proposal = info.get('proposal')
            if isinstance(ref, dict):
                center = ref.get('loc')
                sigma = ref.get('scale', proposal)
            else:
                center = ref
                sigma = proposal

            # Store a plain label without math delimiters.
            label = info.get('latex', name).replace('$', '')

            # Store the metadata needed by the shared BaseLikelihood methods.
            params.append(
                ParameterInfo(
                    name=name,
                    label=label,
                    scale=1.0,
                    lower=prior.get('min'),
                    upper=prior.get('max'),
                    center=center,
                    sigma=sigma
                )
            )

        return params
    
    def loglkl(self, x):
        # Convert the input vector into the parameter dictionary expected by Cobaya.
        position = {param.name: float(value) for param, value in zip(self._params, x)}

        # Evaluate Cobaya's native log-posterior without derived parameters.
        result = self.cobaya_model.logposterior(position, return_derived=False)

        return float(result.logpost)
