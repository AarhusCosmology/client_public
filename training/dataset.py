import heapq
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from scipy.special import gamma, logsumexp


def _unit_ball_volume(dim):
    """Volume of the unit ball in R^dim: V_dim = pi^(dim/2) / Gamma(dim/2 + 1)."""
    return np.pi ** (dim / 2) / gamma(dim / 2 + 1)


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


def _systematic_resample(log_weights, M):
    """Draw M indices from unnormalised log-weights via systematic resampling.

    Compared with multinomial resampling, systematic resampling achieves the
    same O(M) cost with variance lower by a factor of M.
    """
    log_w = log_weights - logsumexp(log_weights)
    w = np.exp(log_w)
    w /= w.sum()  # numerical safety
    cumsum = np.cumsum(w)
    cumsum[-1] = 1.0  # clamp to exactly 1
    u0 = np.random.uniform(0.0, 1.0 / M)
    positions = u0 + np.arange(M) / M
    return np.searchsorted(cumsum, positions)


def _union_log_rho(nn, accepted_arr, query_white, k):
    """Log k-NN density at query points against the union of nn and accepted_arr.

    Queries the training-set index for k neighbours and, if any accepted points
    exist, queries them via brute force. The merged distances are used to find
    the k-th nearest neighbour in the union. query_white must already be in
    the same whitened coordinate system as nn.

    Parameters
    ----------
    nn : _WhitenedKNN
    accepted_arr : np.ndarray, shape (A, dim)  (already whitened; A may be 0)
    query_white : np.ndarray, shape (M, dim)  (already whitened)
    k : int

    Returns
    -------
    log_rho : np.ndarray, shape (M,)
    """
    dim = nn.n_features_in_
    d_train, _ = nn._nn.kneighbors(query_white, n_neighbors=k)  # (M, k)

    A = len(accepted_arr)
    if A > 0:
        diff = query_white[:, None, :] - accepted_arr[None, :, :]  # (M, A, dim)
        d_acc = np.sqrt((diff ** 2).sum(-1))                        # (M, A)
        if A >= k:
            d_acc_top = np.partition(d_acc, k - 1, axis=1)[:, :k]
        else:
            d_acc_top = d_acc  # fewer than k accepted points; use all
        all_d = np.concatenate([d_train, d_acc_top], axis=1)
        r_k = np.partition(all_d, k - 1, axis=1)[:, k - 1]
    else:
        r_k = d_train[:, k - 1]

    return np.log(k / (_unit_ball_volume(dim) * r_k ** dim))


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

    def select_candidates(self, chain, logposts, surrogate, n_augment,
                          sampling_temperature=1.0, pool_factor=20):
        """Select n_augment candidate points from the MCMC chain for true-likelihood evaluation.

        Draws M = pool_factor * n_augment candidates via importance-weighted
        systematic resampling from the chain (biasing toward the training target
        density L^{1/T}), scores each candidate using the surrogate likelihood
        and the current k-NN density estimate, then runs a lazy min-heap with
        incremental union-query density updates to select the n_augment candidates
        that are most under-represented relative to the target density.

        The score of a candidate θ is v(θ) = log ρ(θ) - log L_surr(θ) / T.
        A low score means the point lies in a region that is sparse relative to
        the target density — exactly where new training data is most valuable.

        Monotonicity: adding an accepted point can only increase the local density
        of nearby candidates (more neighbours → smaller r_k → higher ρ → higher
        score). Heap entries therefore never become erroneously low, only
        stale-high, which makes the lazy recomputation scheme correct.

        Parameters
        ----------
        chain : array, shape (S, ndim)
            Flat MCMC chain with burn-in discarded.
        logposts : array, shape (S,)
            Log-posterior values stored by the sampler: log_L(θ_i) / T_MC
            (uniform prior absorbed).
        surrogate : SurrogateLikelihood
        n_augment : int
            Number of candidates to select.
        sampling_temperature : float
            Temperature T_MC at which the chain was sampled.
        pool_factor : int
            Pool size multiplier: M = pool_factor * n_augment. Default 20.

        Returns
        -------
        selected : np.ndarray, shape (n_selected, ndim)
            Selected candidate points ready for true-likelihood evaluation.
            n_selected <= n_augment (may be less if the pool is exhausted).
        """
        chain = np.asarray(chain, dtype=np.float32)
        logposts = np.asarray(logposts, dtype=np.float64)
        S = len(chain)
        M = min(pool_factor * n_augment, S)

        # IS resampling: draw M candidates biased toward the training target
        # density q ∝ L^{1/T} by reweighting the chain from q_MC ∝ L^{1/T_MC}.
        # logposts_i = log_L_i / T_MC, so
        #   log w_i = log_L_i * (1/T - 1/T_MC) = logposts_i * (T_MC/T - 1) * (-1)
        # which simplifies to (T_MC/T - 1) * logposts_i (= 0 when T == T_MC).
        log_weights = (sampling_temperature / self.target_temperature - 1.0) * logposts
        pool_idx = _systematic_resample(log_weights, M)
        candidates = chain[pool_idx]
        print(f"Augmentation: pool of {M} candidates (pool_factor={pool_factor}, "
              f"chain length={S})")

        candidates_white = self._nn.whiten(candidates)
        k = self.n_neighbors
        dim = candidates_white.shape[1]
        ball_vol = _unit_ball_volume(dim)

        # Precompute training-set k-NN distances once for all M candidates (M, k).
        # This eliminates repeated sklearn queries inside the heap loop.
        d_train_all, _ = self._nn._nn.kneighbors(candidates_white, n_neighbors=k)

        # Per-candidate top-k distances to accepted points, maintained incrementally.
        # Initialised to inf so that training-set distances dominate until points are accepted.
        d_acc_topk = np.full((M, k), np.inf, dtype=np.float64)

        log_L_surrs = surrogate.loglkl(tf.constant(candidates, dtype=tf.float32)).numpy().astype(np.float64)

        # Initial bulk scoring (no accepted points yet → d_acc_topk all inf).
        r_k_init = d_train_all[:, k - 1]
        log_rhos_init = np.log(k / (ball_vol * r_k_init ** dim))
        init_scores = log_rhos_init - log_L_surrs / self.target_temperature

        # Build a lazy min-heap of (score, candidate_index).
        heap = [(float(s), i) for i, s in enumerate(init_scores)]
        heapq.heapify(heap)

        selected_indices = []

        while heap and len(selected_indices) < n_augment:
            stored_score, i = heapq.heappop(heap)

            # Recompute score using precomputed training distances and the
            # incrementally maintained accepted-point distances.
            merged_i = np.concatenate([d_train_all[i:i+1], d_acc_topk[i:i+1]], axis=1)
            r_k = np.partition(merged_i, k - 1, axis=1)[0, k - 1]
            new_score = np.log(k / (ball_vol * r_k ** dim)) - log_L_surrs[i] / self.target_temperature

            if new_score > stored_score + 1e-10:
                # Score increased: the entry was stale. Push back with updated score.
                heapq.heappush(heap, (float(new_score), i))
                continue

            # True minimum found — record candidate and update density incrementally.
            selected_indices.append(i)

            d_new = np.sqrt(((candidates_white - candidates_white[i]) ** 2).sum(-1, keepdims=True))
            combined = np.concatenate([d_acc_topk, d_new], axis=1)  # (M, k+1)
            d_acc_topk[:] = np.partition(combined, k - 1, axis=1)[:, :k]

        print(f"  Selected {len(selected_indices)}/{n_augment} candidates for likelihood evaluation")
        return candidates[selected_indices]

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

        n_added = len(new_inputs)
        print(f"  {n_added}/{len(candidates)} points added"
              + (f", {n_nonfinite} discarded as non-finite" if n_nonfinite else ""))

        if n_added > 0:
            self.inputs = np.concatenate([self.inputs, new_inputs])
            self.targets = np.concatenate([
                self.targets,
                new_targets.reshape(-1, 1).astype(np.float32),
            ])
            self._nn = _WhitenedKNN(self.inputs, self.n_neighbors + 1)
