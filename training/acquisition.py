import heapq
import numpy as np
import tensorflow as tf

from scipy.special import gamma, logsumexp

def _unit_ball_volume(dim):
    """Volume of the unit ball in R^dim: V_dim = pi^(dim/2) / Gamma(dim/2 + 1)."""
    return np.pi ** (dim / 2) / gamma(dim / 2 + 1)


def _weighted_sample_without_replacement(log_weights, M):
    """Draw M unique indices from unnormalised log-weights.

    Uses the Gumbel-top-k trick: arg top-k(log w_i + g_i) with i.i.d.
    g_i ~ Gumbel(0, 1) samples exactly from the weighted distribution
    without replacement.
    """
    S = len(log_weights)
    if M <= 0 or S == 0:
        return np.empty((0,), dtype=np.int64)
    if M >= S:
        return np.arange(S, dtype=np.int64)

    log_w = np.asarray(log_weights, dtype=np.float64)
    finite = np.isfinite(log_w)
    if not finite.any():
        return np.random.choice(S, size=M, replace=False)

    # Stabilise finite weights and keep non-finite ones effectively unselectable.
    log_w = log_w.copy()
    log_w[finite] -= np.max(log_w[finite])
    log_w[~finite] = -np.inf

    u = np.random.uniform(np.finfo(np.float64).tiny, 1.0, size=S)
    gumbel = -np.log(-np.log(u))
    keys = log_w + gumbel

    top = np.argpartition(keys, -M)[-M:]
    top = top[np.argsort(keys[top])[::-1]]
    return top.astype(np.int64)


def _validate_temperature(temperature, name):
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive; got {temperature!r}")


def _remove_existing_points(candidates, existing_points):
    """Drop candidates that are exact row-wise duplicates of existing points."""
    existing_points = np.asarray(existing_points, dtype=candidates.dtype)

    if candidates.ndim != 2 or existing_points.ndim != 2:
        raise ValueError("Candidate and existing-point arrays must be two-dimensional")

    if candidates.shape[1] != existing_points.shape[1]:
        raise ValueError("Candidate and existing-point dimensions must match")

    if len(candidates) == 0 or len(existing_points) == 0:
        return candidates

    row_dtype = np.dtype((np.void, candidates.dtype.itemsize * candidates.shape[1]))
    candidate_rows = np.ascontiguousarray(candidates).view(row_dtype).ravel()
    existing_rows = np.ascontiguousarray(existing_points).view(row_dtype).ravel()
    return candidates[~np.isin(candidate_rows, existing_rows)]


def _self_excluded_training_distances(nn, train_white, k):
    """Return k self-excluded training-set neighbour distances for each point."""
    n_train = len(train_white)
    n_query = min(n_train, k + 1)
    dists, indices = nn.kneighbors_white(train_white, n_query)

    d_self_excluded = np.empty((n_train, k), dtype=np.float64)
    row_ids = np.arange(n_train)
    for row in range(n_train):
        keep = indices[row] != row_ids[row]
        row_dists = dists[row][keep]
        if len(row_dists) < k:
            raise ValueError("Unable to construct self-excluded training-set neighbours")
        d_self_excluded[row] = row_dists[:k]

    return d_self_excluded

def select_candidates(dataset, chain, logposts, surrogate, n_augment,
                      sampling_temperature=1.0, pool_factor=20):
    """Select n_augment candidate points from the MCMC chain for true-likelihood evaluation.

    Draws M = pool_factor * n_augment candidates via importance-weighted
    sampling without replacement from the chain (biasing toward the training target
    density L^{1/T}), scores each candidate using the surrogate likelihood
    and the current k-NN density estimate, then runs a lazy min-heap with
    incremental union-query density updates to select the n_augment candidates
    that are most under-represented relative to the target density.

    The score of a candidate θ is v(θ) = log ρ(θ) - log L_surr(θ) / T.
    A low score means the point lies in a region that is sparse relative to
    the target density — exactly where new training data is most valuable.

    As accepted points increase local density, stored heap scores can only
    become stale-low, which makes the lazy recomputation scheme correct.

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
    if n_augment < 0:
        raise ValueError("n_augment must be non-negative")
    if pool_factor <= 0:
        raise ValueError("pool_factor must be positive")
    if dataset.n_neighbors <= 0:
        raise ValueError("n_neighbors must be positive")
    k = dataset.n_neighbors
    target_temperature = dataset.target_temperature
    _validate_temperature(target_temperature, "target_temperature")
    _validate_temperature(sampling_temperature, "sampling_temperature")

    chain = np.asarray(chain, dtype=np.float32)
    logposts = np.asarray(logposts, dtype=np.float64)
    if len(chain) != len(logposts):
        raise ValueError(
            f"chain and logposts must have the same length; got {len(chain)} and {len(logposts)}"
        )
    S = len(chain)
    M = min(pool_factor * n_augment, S)
    n_train = len(dataset.inputs)
    if n_train < 2:
        raise ValueError("Candidate selection requires at least two training points")
    k = min(k, n_train - 1)

    # IS resampling: draw M candidates biased toward the training target
    # density q ∝ L^{1/T} by reweighting the chain from q_MC ∝ L^{1/T_MC}.
    # logposts_i = log_L_i / T_MC, so
    #   log w_i = log_L_i * (1/T - 1/T_MC) = logposts_i * (T_MC/T - 1)
    # which simplifies to (T_MC/T - 1) * logposts_i (= 0 when T == T_MC).
    log_weights = (sampling_temperature / target_temperature - 1.0) * logposts
    pool_idx = _weighted_sample_without_replacement(log_weights, M)
    candidates = chain[pool_idx]

    # Chain indices are unique, but rejected MCMC proposals can repeat states.
    # Keep only unique coordinates so augmentation candidates are point-unique.
    if len(candidates) > 0:
        _, unique_pos = np.unique(candidates, axis=0, return_index=True)
        if len(unique_pos) < len(candidates):
            unique_pos.sort()
            candidates = candidates[unique_pos]

    candidates = _remove_existing_points(candidates, dataset.inputs)

    if len(candidates) == 0:
        print("Augmentation: candidate pool exhausted after duplicate filtering")
        return np.empty((0, chain.shape[1]), dtype=np.float32)

    M = len(candidates)
    print(f"Augmentation: pool of {M} candidates (pool_factor={pool_factor}, "
          f"chain length={S})")

    candidates_white = nn.whiten(candidates)
    dim = candidates_white.shape[1]
    ball_vol = _unit_ball_volume(dim)
    log_k = np.log(k)
    log_ball_vol = np.log(ball_vol)

    # Precompute training-set k-NN distances once for all M candidates (M, k).
    # This eliminates repeated sklearn queries inside the heap loop.
    d_train_all, _ = nn.kneighbors_white(candidates_white, k)

    # Per-candidate top-k distances to accepted points, maintained incrementally.
    # Initialised to inf so that training-set distances dominate until points are accepted.
    d_acc_topk = np.full((M, k), np.inf, dtype=np.float64)

    log_L_surrs = surrogate.loglkl(tf.constant(candidates, dtype=tf.float32)).numpy().astype(np.float64)

    # Initial bulk scoring (no accepted points yet → d_acc_topk all inf).
    r_k_init = d_train_all[:, k - 1]
    r_k_init = np.maximum(r_k_init, 1e-12)
    log_rhos_init = log_k - log_ball_vol - dim * np.log(r_k_init)
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
        r_k = max(r_k, 1e-12)
        new_score = log_k - log_ball_vol - dim * np.log(r_k) - log_L_surrs[i] / target_temperature

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


# ---------------------------------------------------------------------------
# True-likelihood acceptance gate
# ---------------------------------------------------------------------------

def _compute_log_c(d_train_self, log_L_train, d_acc_topk_train,
                   accepted_indices, log_L_true,
                   d_train_cand, d_cand_pairwise,
                   k, T, ball_vol, dim):
    """Compute the normalization-based log c over the current union.

    Training-set and accepted-buffer points both contribute to the current
    union. For each point we compute its true log-likelihood and current
    self-excluded log density, then evaluate

        log c = log N_union - logsumexp(log L_i / T - log rho_i).

    Self-neighbours are excluded in both the training-set and accepted-buffer
    density calculations.

    Parameters
    ----------
    d_train_self : array, shape (N_train, k)
        Top-k distances from each training set point to its k nearest training
        set neighbours, with self excluded (distance 0 removed).
    log_L_train : array, shape (N_train,)
        True log-likelihood values for training set points.
    d_acc_topk_train : array, shape (N_train, k)
        Top-k distances from each training set point to the accepted buffer.
        Entries are inf where the buffer is empty or has fewer than k points.
    accepted_indices : list[int]
        Indices of accepted candidates inside the candidate arrays.
    log_L_true : array, shape (N_candidates,)
        True log-likelihood values for all finite candidates.
    d_train_cand : array, shape (N_candidates, k)
        Precomputed candidate-to-training k-NN distances.
    d_cand_pairwise : array, shape (N_candidates, N_candidates)
        Precomputed candidate-to-candidate distance matrix in whitened space.
    k, T, ball_vol, dim : scalar
        Density estimator parameters.

    Returns
    -------
    log_c : float
        Normalization constant for the acceptance criterion.
    """
    log_k = np.log(k)
    log_ball_vol = np.log(ball_vol)

    # --- Training set (vectorised) ---
    merged = np.concatenate([d_train_self, d_acc_topk_train], axis=1)  # (N_train, 2k)
    r_k = np.partition(merged, k - 1, axis=1)[:, k - 1]               # (N_train,)
    r_k = np.maximum(r_k, 1e-12)
    train_log_rho = log_k - log_ball_vol - dim * np.log(r_k)

    union_log_L = [np.asarray(log_L_train, dtype=np.float64)]
    union_log_rho = [train_log_rho]

    # --- Accepted buffer ---
    n_acc = len(accepted_indices)
    if n_acc > 0:
        idx = np.asarray(accepted_indices, dtype=np.int64)
        acc_log_L = log_L_true[idx]  # (n_acc,)
        d_to_train = d_train_cand[idx].astype(np.float64)  # (n_acc, k)
        d_to_acc = np.full((n_acc, k), np.inf, dtype=np.float64)

        if n_acc > 1:
            d_pair = d_cand_pairwise[np.ix_(idx, idx)].copy()
            np.fill_diagonal(d_pair, np.inf)
            top = min(k, n_acc - 1)
            d_to_acc[:, :top] = np.partition(d_pair, top - 1, axis=1)[:, :top]

        merged_acc = np.concatenate([d_to_train, d_to_acc], axis=1)
        r_k_acc = np.partition(merged_acc, k - 1, axis=1)[:, k - 1]
        r_k_acc = np.maximum(r_k_acc, 1e-12)
        acc_log_rho = log_k - log_ball_vol - dim * np.log(r_k_acc)

        union_log_L.append(acc_log_L)
        union_log_rho.append(acc_log_rho)

    union_log_L = np.concatenate(union_log_L)
    union_log_rho = np.concatenate(union_log_rho)
    return float(np.log(len(union_log_L)) - logsumexp(union_log_L / T - union_log_rho))


def apply_true_gate(dataset, candidates, log_L_true):
    """Apply a true-likelihood acceptance gate to surrogate-selected candidates.

    Builds a separate true-score priority queue (the surrogate heap is
    discarded) and accepts candidates whose density-adjusted true score falls
    below the dynamic threshold log_c, computed over the union of the current
    training set and the accepted points so far.  Surrogate and true scores
    are never mixed.

    Algorithm
    ---------
    1. Discard non-finite candidates.
    2. Whiten the remaining candidates using the FIXED training-set transform.
    3. Build a lazy min-heap with scores
           v_true(θ) = log ρ(θ) - log L_true(θ) / T
       where ρ is the k-NN density in the union of training set and accepted
       buffer.
     4. Compute an initial normalization-based log_c over the training set
         (with self-excluded k-NN density).
    5. Pop the lowest-score candidate, recompute its score (density may have
       increased due to earlier acceptances), push it back if stale.
     6. Accept the candidate only if its true score ≤ log_c; if the current
         minimum fails, stop because all remaining candidates must also fail.
    7. After each acceptance: update the accepted buffer and the incremental
       top-k distance arrays, then recompute log_c over the full union
       (training set + accepted buffer, both with self-exclusion).

    Parameters
    ----------
    dataset : TrainingDataset
        k-NN index and whitening transform must reflect the *current* training
        set (not yet updated with the new candidates).
    candidates : np.ndarray, shape (N, ndim)
        Candidates previously selected by ``select_candidates``.
    log_L_true : np.ndarray, shape (N,)
        True log-likelihood values returned by the parallel evaluation step.

    Returns
    -------
    accepted_candidates : np.ndarray, shape (n_accepted, ndim)
    accepted_log_L : np.ndarray, shape (n_accepted,)
        n_accepted ≤ N; may be strictly smaller when some candidates lie in
        already well-covered regions.
    """
    nn = dataset.nn
    T = dataset.target_temperature
    _validate_temperature(T, "target_temperature")

    candidates = np.asarray(candidates, dtype=np.float32)
    log_L_true = np.asarray(log_L_true, dtype=np.float64)
    if len(candidates) != len(log_L_true):
        raise ValueError(
            f"candidates and log_L_true must have the same length; got {len(candidates)} and {len(log_L_true)}"
        )

    # Step 1: Remove non-finite candidates.
    valid = np.isfinite(log_L_true)
    n_nonfinite = int((~valid).sum())
    if n_nonfinite:
        print(f"  True gate: {n_nonfinite} non-finite candidates removed")
    candidates = candidates[valid]
    log_L_true = log_L_true[valid]
    N = len(candidates)

    if N == 0:
        print("  True gate: no finite candidates to evaluate")
        return candidates, log_L_true

    dim = candidates.shape[1]
    ball_vol = _unit_ball_volume(dim)
    log_ball_vol = np.log(ball_vol)

    # Step 2: Whiten candidates using the FIXED training-set transform.
    cand_white = nn.whiten(candidates).astype(np.float64)

    # --- Fixed training-set quantities (computed once, never updated) ---
    train_white = nn.whiten(dataset.inputs).astype(np.float64)
    N_train = len(train_white)
    if N_train < 2:
        raise ValueError("True-likelihood gate requires at least two training points")
    k = min(dataset.n_neighbors, N_train - 1)
    log_k = np.log(k)

    # The whitening transform and training-set index stay fixed throughout
    # this acceptance round; they are only rebuilt after accepted points are
    # appended to the permanent dataset.

    # Query one extra neighbour, then remove the exact self index explicitly.
    d_train_self = _self_excluded_training_distances(nn, train_white, k)

    log_L_train = dataset.targets[:, 0].astype(np.float64)    # (N_train,)

    # k-NN distances from candidates to the fixed training set (no self-issue).
    d_train_cand, _ = nn.kneighbors_white(cand_white, k)
    d_train_cand = d_train_cand.astype(np.float64)             # (N, k)

    # Precompute candidate-to-candidate distances once (whitened space).
    d_cand_pairwise = np.sqrt(
        ((cand_white[:, np.newaxis, :] - cand_white[np.newaxis, :, :]) ** 2).sum(-1)
    ).astype(np.float64)

    # --- Incremental accepted-buffer distance buffers ---
    # Top-k distances from each TRAINING SET point to the accepted buffer.
    d_acc_topk_train = np.full((N_train, k), np.inf, dtype=np.float64)
    # Top-k distances from each CANDIDATE to the accepted buffer.
    d_acc_topk_cand = np.full((N, k), np.inf, dtype=np.float64)

    # Step 3: Build true-score min-heap (empty accepted buffer → use training distances).
    r_k_init = d_train_cand[:, k - 1]
    r_k_init = np.maximum(r_k_init, 1e-12)
    log_rho_init = log_k - log_ball_vol - dim * np.log(r_k_init)
    scores_init = log_rho_init - log_L_true / T

    heap = [(float(s), i) for i, s in enumerate(scores_init)]
    heapq.heapify(heap)

    # Steps 5–7 track accepted candidates by index into candidate arrays.
    accepted_indices = []

    # Step 4: Initial log_c (empty accepted buffer → union = training set).
    log_c = _compute_log_c(
        d_train_self, log_L_train, d_acc_topk_train,
        accepted_indices, log_L_true,
        d_train_cand, d_cand_pairwise,
        k, T, ball_vol, dim,
    )
    print(f"  True gate: initial log_c = {log_c:.4f}, evaluating {N} candidates")

    # Steps 5–7: Lazy acceptance loop.

    while heap:
        stored_score, i = heapq.heappop(heap)

        # Recompute true score for candidate i using current union density.
        merged_i = np.concatenate(
            [d_train_cand[i:i+1], d_acc_topk_cand[i:i+1]], axis=1
        )  # (1, 2k)
        r_k_i = np.partition(merged_i, k - 1, axis=1)[0, k - 1]
        r_k_i = max(r_k_i, 1e-12)
        new_score = log_k - log_ball_vol - dim * np.log(r_k_i) - log_L_true[i] / T

        if new_score > stored_score + 1e-10:
            # Entry was stale (density increased since last push); requeue.
            heapq.heappush(heap, (float(new_score), i))
            continue

        # True minimum for this candidate.  Apply acceptance gate.
        if new_score > log_c:
            # The valid minimum already exceeds the threshold, so every
            # remaining candidate must also fail.
            break

        # --- Accept ---
        accepted_indices.append(i)
        new_white = cand_white[i]

        # Incrementally update top-k accepted distances for all candidates.
        d_new_cand = d_cand_pairwise[:, i]  # (N,)
        combined_cand = np.concatenate(
            [d_acc_topk_cand, d_new_cand[:, np.newaxis]], axis=1
        )  # (N, k+1)
        d_acc_topk_cand[:] = np.partition(combined_cand, k - 1, axis=1)[:, :k]

        # Incrementally update top-k accepted distances for training set points.
        d_new_train = np.sqrt(((train_white - new_white) ** 2).sum(-1))  # (N_train,)
        combined_train = np.concatenate(
            [d_acc_topk_train, d_new_train[:, np.newaxis]], axis=1
        )  # (N_train, k+1)
        d_acc_topk_train[:] = np.partition(combined_train, k - 1, axis=1)[:, :k]

        # Recompute log_c over updated union (training set + accepted buffer).
        log_c = _compute_log_c(
            d_train_self, log_L_train, d_acc_topk_train,
            accepted_indices, log_L_true,
            d_train_cand, d_cand_pairwise,
            k, T, ball_vol, dim,
        )

    n_accepted = len(accepted_indices)
    print(
        f"  True gate: accepted {n_accepted}/{N} candidates "
        f"(final log_c = {log_c:.4f})"
    )

    if n_accepted == 0:
        return np.empty((0, dim), dtype=np.float32), np.empty(0, dtype=np.float64)

    return candidates[accepted_indices], log_L_true[accepted_indices]
