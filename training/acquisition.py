import heapq
import numpy as np
import tensorflow as tf
from scipy.special import gamma, logsumexp


def _unit_ball_volume(dim):
    """Volume of the unit ball in R^dim: V_dim = pi^(dim/2) / Gamma(dim/2 + 1)."""
    return np.pi ** (dim / 2) / gamma(dim / 2 + 1)


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


def select_candidates(dataset, chain, logposts, surrogate, n_augment,
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
    dataset : TrainingDataset
        Provides the whitened k-NN density index (``dataset.nn``), the number of
        neighbours (``dataset.n_neighbors``) and the target temperature
        (``dataset.target_temperature``).
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
    nn = dataset.nn
    k = dataset.n_neighbors
    target_temperature = dataset.target_temperature

    chain = np.asarray(chain, dtype=np.float32)
    logposts = np.asarray(logposts, dtype=np.float64)
    S = len(chain)
    M = min(pool_factor * n_augment, S)

    # IS resampling: draw M candidates biased toward the training target
    # density q ∝ L^{1/T} by reweighting the chain from q_MC ∝ L^{1/T_MC}.
    # logposts_i = log_L_i / T_MC, so
    #   log w_i = log_L_i * (1/T - 1/T_MC) = logposts_i * (T_MC/T - 1) * (-1)
    # which simplifies to (T_MC/T - 1) * logposts_i (= 0 when T == T_MC).
    log_weights = (sampling_temperature / target_temperature - 1.0) * logposts
    pool_idx = _systematic_resample(log_weights, M)
    candidates = chain[pool_idx]
    print(f"Augmentation: pool of {M} candidates (pool_factor={pool_factor}, "
          f"chain length={S})")

    candidates_white = nn.whiten(candidates)
    dim = candidates_white.shape[1]
    ball_vol = _unit_ball_volume(dim)

    # Precompute training-set k-NN distances once for all M candidates (M, k).
    # This eliminates repeated sklearn queries inside the heap loop.
    d_train_all, _ = nn.kneighbors_white(candidates_white, k)

    # Per-candidate top-k distances to accepted points, maintained incrementally.
    # Initialised to inf so that training-set distances dominate until points are accepted.
    d_acc_topk = np.full((M, k), np.inf, dtype=np.float64)

    log_L_surrs = surrogate.loglkl(tf.constant(candidates, dtype=tf.float32)).numpy().astype(np.float64)

    # Initial bulk scoring (no accepted points yet → d_acc_topk all inf).
    r_k_init = d_train_all[:, k - 1]
    log_rhos_init = np.log(k / (ball_vol * r_k_init ** dim))
    init_scores = log_rhos_init - log_L_surrs / target_temperature

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
        new_score = np.log(k / (ball_vol * r_k ** dim)) - log_L_surrs[i] / target_temperature

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
