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

def select_candidates(dataset, chain, logposts, surrogate, batch_size,
                      sampling_temperature=1.0, pool_factor=20):
    """Select ``batch_size`` candidate points from the MCMC chain for true-likelihood evaluation.

    Implements the density-deficit selection rule. Candidates are drawn from
    the chain (importance-reweighted to the training target density
    q_surr ∝ L_surr^{1/T}) and then thinned so that the accepted points are
    distributed in proportion to the *positive point-number deficit*

        d_+(θ) = [ N_{t+1} q_surr(θ) - ρ(θ) ]_+ ,   N_{t+1} = N_t + batch_size,

    i.e. they preferentially fill regions that are under-represented relative
    to where the current surrogate target density says training points should
    be. Each candidate is retained with the bounded density-deficit probability

        a(θ) = [ 1 - ρ(θ) / (N_{t+1} q_surr(θ)) ]_+ = [ 1 - 1/D(θ) ]_+ ,
        D(θ) = N_{t+1} q_surr(θ) / ρ(θ) ,

    and points are drawn one at a time (categorical ∝ a) with the candidate
    density ρ updated after every acceptance. This sequential update is what
    distinguishes the rule from a one-shot draw: once a sparse pocket receives
    a point its local deficit shrinks, so it stops attracting the next one and
    the realised training density tracks N_{t+1} q_surr faithfully even when
    ``batch_size`` is an appreciable fraction of N_t.

    Whitening. All densities and distances are computed in the whitened
    (Mahalanobis) coordinates provided by ``dataset.nn``; the surrogate is
    evaluated in the original coordinates. The ratio D — and hence a — is
    invariant under whitening (the √det Σ factors of q_surr and ρ cancel), so
    no Jacobian correction is applied. The k-NN density uses the bias-corrected
    (k-1) estimator for training points (leave-one-out) and the k estimator for
    candidate query points.

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
    batch_size : int
        Number of candidates to select (B_t).
    sampling_temperature : float
        Temperature T_MC at which the chain was sampled.
    pool_factor : int
        Pool size multiplier: M = pool_factor * batch_size. Default 20.

    Returns
    -------
    selected : np.ndarray, shape (n_selected, ndim)
        Selected candidate points ready for true-likelihood evaluation.
        n_selected <= batch_size (may be less if every remaining candidate
        already lies in an over-dense region).
    """
    nn = dataset.nn
    k = dataset.n_neighbors
    target_temperature = dataset.target_temperature
    n_train = len(dataset.inputs)
    N_next = n_train + batch_size

    chain = np.asarray(chain, dtype=np.float32)
    logposts = np.asarray(logposts, dtype=np.float64)
    S = len(chain)
    M = min(pool_factor * batch_size, S)

    # IS resampling: draw M candidates biased toward the training target
    # density q ∝ L^{1/T} by reweighting the chain from q_MC ∝ L^{1/T_MC}.
    # logposts_i = log_L_i / T_MC, so
    #   log w_i = log_L_i * (1/T - 1/T_MC) = (T_MC/T - 1) * logposts_i
    # which vanishes when T == T_MC (the chain already samples q_surr).
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

    M = len(candidates)
    print(f"Augmentation: pool of {M} candidates (pool_factor={pool_factor}, "
          f"chain length={S})")

    candidates_white = nn.whiten(candidates)
    dim = candidates_white.shape[1]
    log_ball_vol = np.log(_unit_ball_volume(dim))

    # ---- Surrogate target-density normalisation log Ẑ_surr (frozen per call) ----
    # Estimated once over the training points in whitened coordinates:
    #   log Ẑ_surr = logsumexp_j [ ℓ_surr(θ_j)/T - log ρ̃_j ],
    # with the leave-one-out (k-1) density ρ̃_j = (k-1)/(V_n r_{k,j}^n) where
    # r_{k,j} is the distance to the j-th point's k-th nearest *other* training
    # point. The index is fitted with n_neighbors+1 neighbours, so querying k+1
    # returns [self (d=0), 1st, …, k-th other]; column k is the k-th other.
    train_white = nn.whiten(dataset.inputs)
    d_train_self, _ = nn.kneighbors_white(train_white, k + 1)
    r_k_train = d_train_self[:, k]
    log_rho_train = np.log(k - 1) - log_ball_vol - dim * np.log(r_k_train)
    log_L_surr_train = surrogate.loglkl(
        tf.constant(dataset.inputs, dtype=tf.float32)).numpy().astype(np.float64)
    log_Z_surr = logsumexp(log_L_surr_train / target_temperature - log_rho_train)

    # ---- Per-candidate surrogate log-likelihood and constant log-q_surr offset ----
    log_L_surrs = surrogate.loglkl(
        tf.constant(candidates, dtype=tf.float32)).numpy().astype(np.float64)
    # log[ N_{t+1} q_surr(θ) ] = log N_{t+1} + ℓ_surr/T - log Ẑ_surr  (whitened).
    log_target = np.log(N_next) + log_L_surrs / target_temperature - log_Z_surr

    # Candidate density uses the k estimator against the union (training set +
    # accepted points). d_train_all holds the k nearest *training* distances;
    # d_acc_topk holds the k nearest *accepted-point* distances (∞ until any
    # point is accepted). r_k is the k-th smallest of their union.
    d_train_all, _ = nn.kneighbors_white(candidates_white, k)
    d_acc_topk = np.full((M, k), np.inf, dtype=np.float64)
    log_k = np.log(k)

    selected_indices = []
    available = np.ones(M, dtype=bool)

    for _ in range(batch_size):
        # Current candidate density ρ_cur over the union (whitened, k estimator).
        merged = np.concatenate([d_train_all, d_acc_topk], axis=1)  # (M, 2k)
        r_k = np.partition(merged, k - 1, axis=1)[:, k - 1]
        log_rho_cur = log_k - log_ball_vol - dim * np.log(r_k)

        # log D = log[N_{t+1} q_surr] - log ρ_cur ; a = [1 - 1/D]_+ = [1 - e^{-logD}]_+.
        log_D = log_target - log_rho_cur
        a = np.where(log_D > 0.0, -np.expm1(-log_D), 0.0)
        a[~available] = 0.0

        if not np.any(a > 0.0):
            # Every remaining candidate is already at/above the target density.
            print("  Density deficit exhausted: no under-represented candidates remain")
            break

        # Categorical draw ∝ a via the Gumbel-max trick (log a + Gumbel noise).
        log_a = np.full(M, -np.inf)
        np.log(a, out=log_a, where=a > 0.0)
        u = np.random.uniform(np.finfo(np.float64).tiny, 1.0, size=M)
        gumbel = -np.log(-np.log(u))
        i = int(np.argmax(log_a + gumbel))

        selected_indices.append(i)
        available[i] = False

        # Fold the newly accepted point into every candidate's accepted-distance
        # buffer, keeping the k smallest — this is the sequential density update.
        d_new = np.sqrt(((candidates_white - candidates_white[i]) ** 2).sum(-1, keepdims=True))
        combined = np.concatenate([d_acc_topk, d_new], axis=1)  # (M, k+1)
        d_acc_topk = np.partition(combined, k - 1, axis=1)[:, :k]
        d_acc_topk[i] = np.inf  # an accepted candidate is no longer a query target

    print(f"  Selected {len(selected_indices)}/{batch_size} candidates for likelihood evaluation")
    return candidates[selected_indices]
