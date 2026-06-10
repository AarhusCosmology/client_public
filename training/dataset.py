import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.neighbors import NearestNeighbors

class _WhitenedKNN:
    """k-NN index operating in whitened (Euclidean) space.

    Pre-computes the whitening transform W = L^{-T} (where Sigma = L L^T is
    the Cholesky decomposition of the sample covariance) and fits an index on
    the whitened data using sklearn's algorithm='auto' heuristic (KD-tree,
    ball tree, or brute force depending on dimensionality and dataset size).
    Distances in whitened space are identical to Mahalanobis distances.
    """
    def __init__(self, points, n_neighbors):
        self.n_features_in_ = points.shape[1]
        self._mean = points.mean(axis=0)
        cov = np.cov(points, rowvar=False)
        if cov.ndim == 0:
            cov = cov.reshape(1, 1)
        # Small jitter for numerical stability if any parameter has near-zero variance.
        cov = cov + 1e-10 * np.eye(cov.shape[0])
        L = np.linalg.cholesky(cov)
        # W = L^{-T}: x_white = (x - mean) @ W  →  ||x_white - y_white||_2 == mahalanobis(x, y)
        self._W = np.linalg.solve(L, np.eye(L.shape[0])).T
        self._nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', metric='euclidean')
        self._nn.fit(self.whiten(points))

    def whiten(self, X):
        """Apply the whitening transform: z = (x - μ) @ W."""
        return (X - self._mean) @ self._W

    def kneighbors(self, X):
        return self._nn.kneighbors(self.whiten(X))

    def kneighbors_white(self, X_white, n_neighbors):
        """k-NN query for points already expressed in whitened coordinates."""
        return self._nn.kneighbors(X_white, n_neighbors=n_neighbors)


class TrainingDataset:
    def __init__(self, inputs, targets, likelihood, n_neighbors, target_temperature=1.0):
        """
        Parameters
        ----------
        inputs : array, shape (N, ndim)
        targets : array, shape (N, 1)
            Log-likelihood values.
        likelihood : BaseLikelihood
        n_neighbors : int
            Number of neighbours for the k-NN density estimator.
        target_temperature : float
            Temperature T defining the target distribution L^{1/T} * pi
            that the dataset is designed to cover.
        """
        self.inputs = inputs.copy()
        self.targets = targets.copy()
        self.likelihood = likelihood
        self.names = likelihood.get_param_names()
        self.n_neighbors = n_neighbors
        self.target_temperature = target_temperature
        self.iteration = None
        self._nn = _WhitenedKNN(self.inputs, n_neighbors + 1)

    @property
    def nn(self):
        """Whitened k-NN density index over the current training inputs."""
        return self._nn

    def save(self, path):
        df = pd.DataFrame(self.inputs, columns=self.names)
        df['loglkl'] = self.targets[:, 0]
        df.to_csv(path, index=False)

    @classmethod
    def load(cls, training_data_dir, likelihood, n_neighbors, target_temperature=1.0,
             iteration=None):
        if iteration is None:
            iteration = max(
                int(p.stem.split('_it_')[1])
                for p in Path(training_data_dir).glob('training_data_it_*.csv')
            )
        path = Path(training_data_dir) / f'training_data_it_{iteration}.csv'
        df = pd.read_csv(path)
        names = likelihood.get_param_names()
        inputs = df[names].to_numpy(dtype=np.float32)
        targets = df[['loglkl']].to_numpy(dtype=np.float32)
        dataset = cls(inputs, targets, likelihood, n_neighbors, target_temperature)
        dataset.iteration = iteration
        return dataset

    def add_evaluated_points(self, candidates, log_L_values):
        """Append true-likelihood-evaluated points to the dataset.

        Parameters
        ----------
        candidates : array, shape (N, ndim)
            Candidate points returned by select_candidates.
        log_L_values : array, shape (N,)
            True log-likelihood values for each candidate.
        """
        candidates = np.asarray(candidates, dtype=np.float32)
        log_L_values = np.asarray(log_L_values, dtype=np.float64)

        valid = np.isfinite(log_L_values)
        n_nonfinite = int((~valid).sum())
        new_inputs = candidates[valid]
        new_targets = log_L_values[valid]

        n_duplicate_batch = 0
        n_duplicate_existing = 0

        if len(new_inputs) > 0:
            # Keep first occurrence order when removing duplicates in this batch.
            _, unique_idx = np.unique(new_inputs, axis=0, return_index=True)
            if len(unique_idx) < len(new_inputs):
                n_duplicate_batch = len(new_inputs) - len(unique_idx)
                keep = np.sort(unique_idx)
                new_inputs = new_inputs[keep]
                new_targets = new_targets[keep]

        if len(new_inputs) > 0:
            row_dtype = np.dtype((np.void, new_inputs.dtype.itemsize * new_inputs.shape[1]))
            existing_rows = np.ascontiguousarray(self.inputs).view(row_dtype).ravel()
            candidate_rows = np.ascontiguousarray(new_inputs).view(row_dtype).ravel()
            is_new = ~np.isin(candidate_rows, existing_rows)
            n_duplicate_existing = int((~is_new).sum())
            new_inputs = new_inputs[is_new]
            new_targets = new_targets[is_new]

        n_added = len(new_inputs)
        msg = f"  {n_added}/{len(candidates)} points added"
        if n_nonfinite:
            msg += f", {n_nonfinite} discarded as non-finite"
        if n_duplicate_batch:
            msg += f", {n_duplicate_batch} duplicate candidates dropped"
        if n_duplicate_existing:
            msg += f", {n_duplicate_existing} already in training set"
        print(msg)

        if n_added > 0:
            self.inputs = np.concatenate([self.inputs, new_inputs])
            self.targets = np.concatenate([
                self.targets,
                new_targets.reshape(-1, 1).astype(np.float32),
            ])
            self._nn = _WhitenedKNN(self.inputs, self.n_neighbors + 1)