import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.neighbors import NearestNeighbors

class _WhitenedKNN:
    def __init__(self, points, n_neighbors):
        self.ndim = points.shape[1]
        self.n_neighbors = n_neighbors
        self._mean = points.mean(axis=0)
        cov = np.cov(points, rowvar=False)
        
        # Apply small jitter for numerical stability
        # if any parameter has near-zero variance
        cov = cov + 1e-10 * np.eye(self.ndim)

        # Compute the Cholesky factor L where Σ = LL^T
        L = np.linalg.cholesky(cov)
        
        # For a row vector x, the whitening transform can be written as z = (x - μ) W, where W = L^{-T}
        self._W = np.linalg.solve(L, np.eye(self.ndim)).T

        # Build a Euclidean nearest-neighbor index in whitened space
        # (e.g. a tree, or stored points for brute-force search depending on the algorithm)
        self._knn_index = NearestNeighbors(
            n_neighbors=n_neighbors, 
            algorithm='auto',
            metric='euclidean'
        )
        self._knn_index.fit(self.whiten(points))

    def whiten(self, points):
        return (points - self._mean) @ self._W
    
    def query(self, points, n_neighbors=None):
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        return self._knn_index.kneighbors(self.whiten(points), n_neighbors=n_neighbors)
    
class TrainingDataset:
    def __init__(self, inputs, targets, likelihood, n_neighbors, target_temperature):
        self.inputs = np.asarray(inputs, dtype=np.float32).copy()
        self.targets = np.asarray(targets, dtype=np.float32).reshape(-1, 1).copy()
        self.likelihood = likelihood
        self.param_names = likelihood.param_names
        self.n_neighbors = n_neighbors
        self.target_temperature = target_temperature
        self.iteration = None
        self._knn_index = _WhitenedKNN(self.inputs, n_neighbors)

    @property
    def knn_index(self):
        return self._knn_index
    
    def save(self, path):
        df = pd.DataFrame(self.inputs, columns=self.param_names)
        df['loglkl'] = self.targets[:, 0]
        df.to_csv(path, index=False)

    @classmethod
    def load(cls, training_data_dir, likelihood, n_neighbors, target_temperature, iteration=None):
        # If no iteration is specified, load the latest available
        # data_it_<iteration>.csv file.
        if iteration is None:
            iteration = max(
                int(p.stem.rsplit('_', 1)[1])
                for p in Path(training_data_dir).iterdir()
            )
        path = Path(training_data_dir) / f'data_it_{iteration}.csv'
        df = pd.read_csv(path)
        param_names = likelihood.param_names
        inputs = df[param_names].to_numpy(dtype=np.float32)
        targets = df[['loglkl']].to_numpy(dtype=np.float32)
        dataset = cls(inputs, targets, likelihood, n_neighbors, target_temperature)
        dataset.iteration = iteration
        return dataset
    
    def add_data(self, inputs, targets):
        inputs = np.asarray(inputs, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32).reshape(-1, 1)
        self.inputs = np.concatenate([self.inputs, inputs])
        self.targets = np.concatenate([self.targets, targets])
        self._knn_index = _WhitenedKNN(self.inputs, self.n_neighbors)
