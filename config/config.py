import yaml

from dataclasses import dataclass

@dataclass(frozen=True)
class LikelihoodConfig:
    wrapper: str
    input: str

    @classmethod
    def from_dict(cls, d):
        return cls(
            wrapper=str(d['wrapper']),
            input=str(d['input']),
        )

@dataclass(frozen=True)
class PriorConfig:
    n_samples: int
    sampling_strategy: str
    n_sigma: float | None

    @classmethod
    def from_dict(cls, d):
        n_sigma = d.get('n_sigma')
        return cls(
            n_samples=int(d['n_samples']),
            sampling_strategy=str(d['sampling_strategy']),
            n_sigma=None if n_sigma is None else float(n_sigma),
        )


@dataclass(frozen=True)
class AcquisitionConfig:
    n_append: int
    n_neighbors: int
    target_temperature: float
    pool_factor: int

    @classmethod
    def from_dict(cls, d):
        return cls(
            n_append=int(d['n_append']),
            n_neighbors=int(d['n_neighbors']),
            target_temperature=float(d['target_temperature']),
            pool_factor=int(d['pool_factor']),
        )

@dataclass(frozen=True)
class ModelConfig:
    n_layers: int
    n_neurons: int
    activation: str

    @classmethod
    def from_dict(cls, d):
        return cls(
            n_layers=int(d['n_layers']),
            n_neurons=int(d['n_neurons']),
            activation=str(d['activation']),
        )

@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float
    loss: str
    kappa_sigma: float
    n_epochs: int
    batch_size: int
    validation_split: float
    patience: int

    @classmethod
    def from_dict(cls, d):
        return cls(
            learning_rate=float(d['learning_rate']),
            loss=str(d['loss']),
            kappa_sigma=float(d['kappa_sigma']),
            n_epochs=int(d['n_epochs']),
            batch_size=int(d['batch_size']),
            validation_split=float(d['validation_split']),
            patience=int(d['patience']),
        )

@dataclass(frozen=True)
class SamplingConfig:
    sampler: str
    temperature: float
    n_walkers: int
    burn_in: int
    n_steps: int

    @classmethod
    def from_dict(cls, d):
        return cls(
            sampler=str(d['sampler']),
            temperature=float(d['temperature']),
            n_walkers=int(d['n_walkers']),
            burn_in=int(d['burn_in']),
            n_steps=int(d['n_steps']),
        )

@dataclass(frozen=True)
class ConvergenceConfig:
    threshold: float
    metric: str
    max_iterations: int

    @classmethod
    def from_dict(cls, d):
        return cls(
            threshold=float(d['threshold']),
            metric=str(d['metric']),
            max_iterations=int(d['max_iterations']),
        )

@dataclass(frozen=True)
class Config:
    likelihood: LikelihoodConfig
    prior: PriorConfig
    acquisition: AcquisitionConfig
    model: ModelConfig
    training: TrainingConfig
    sampling: SamplingConfig
    convergence: ConvergenceConfig

    @classmethod
    def from_dict(cls, d):
        return cls(
            likelihood=LikelihoodConfig.from_dict(d['likelihood']),
            prior=PriorConfig.from_dict(d['prior']),
            acquisition=AcquisitionConfig.from_dict(d['acquisition']),
            model=ModelConfig.from_dict(d['model']),
            training=TrainingConfig.from_dict(d['training']),
            sampling=SamplingConfig.from_dict(d['sampling']),
            convergence=ConvergenceConfig.from_dict(d['convergence']),
        )

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))
