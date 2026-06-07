import yaml

from dataclasses import dataclass
from pathlib import Path

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
class DataConfig:
    n_samples: int
    n_sigma: float
    n_augment: int
    n_neighbors: int
    target_temperature: float
    pool_factor: int

    @classmethod
    def from_dict(cls, d):
        return cls(
            n_samples=int(d['n_samples']),
            n_sigma=float(d['n_sigma']),
            n_augment=int(d['n_augment']),
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
    n_chains: int
    burn_in: int
    max_steps: int

    @classmethod
    def from_dict(cls, d):
        return cls(
            sampler=str(d['sampler']),
            temperature=float(d['temperature']),
            n_walkers=int(d['n_walkers']),
            n_chains=int(d['n_chains']),
            burn_in=int(d['burn_in']),
            max_steps=int(d['max_steps']),
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
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    sampling: SamplingConfig
    convergence: ConvergenceConfig

    @classmethod
    def from_dict(cls, d):
        return cls(
            likelihood=LikelihoodConfig.from_dict(d['likelihood']),
            data=DataConfig.from_dict(d['data']),
            model=ModelConfig.from_dict(d['model']),
            training=TrainingConfig.from_dict(d['training']),
            sampling=SamplingConfig.from_dict(d['sampling']),
            convergence=ConvergenceConfig.from_dict(d['convergence']),
        )

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))
