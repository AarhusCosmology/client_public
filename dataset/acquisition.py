import numpy as np
from scipy.special import gammaln, logsumexp

from dataset.dataset import _WhitenedKNN


def _log_unit_ball_volume(ndim):
    return (ndim / 2.0) * np.log(np.pi) - gammaln(ndim / 2.0 + 1.0)


def _deduplicate(chain, logposts):
    chain = np.asarray(chain)
    logposts = np.asarray(logposts)
    points = []
    point_logposts = []
    counts = []
    for walker in range(chain.shape[1]):
        x = chain[:, walker, :]
        y = logposts[:, walker]
        starts = np.r_[True, np.any(x[1:] != x[:-1], axis=1)]
        start_indices = np.flatnonzero(starts)
        points.append(x[start_indices])
        point_logposts.append(y[start_indices])
        counts.append(np.diff(np.r_[start_indices, len(x)]))

    return (
        np.concatenate(points, axis=0),
        np.concatenate(point_logposts, axis=0),
        np.concatenate(counts, axis=0),
    )


def select_points(dataset, chain, logposts, n_append, mcmc_temperature, pool_factor):
    knn_index = dataset.knn_index
    ndim = knn_index.ndim
    n_neighbors = dataset.n_neighbors
    target_temperature = dataset.target_temperature
    n_current = len(dataset.inputs)

    # log w_i ∝ (1/T - 1/T_MC) * loglkls_i
    # Since logposts = 1/T_MC * loglkls, we have
    # log w_i ∝ (T_MC/T - 1) * logposts_i
    log_weight_coeff = mcmc_temperature / target_temperature - 1.0

    # Collapse exact duplicate MCMC states, but preserve their total probability mass
    chain_unique, logposts_unique, counts = _deduplicate(chain, logposts)
    log_weights_unique = log_weight_coeff * logposts_unique + np.log(counts)

    # Normalize the unique-point weights
    log_weights_unique = log_weights_unique - logsumexp(log_weights_unique)
    weights = np.exp(log_weights_unique)

    # Draw the candidate pool and a disjoint reference sample in a single weighted
    # draw without replacement. Both are then samples from the target density
    # q ∝ L^(1/T); drawing them disjointly keeps a pool point from turning up as its
    # own nearest neighbor in the reference index below.
    n_available = int(np.count_nonzero(weights))
    pool_size = min(pool_factor * n_append, n_available)
    n_reference = min(n_current, n_available - pool_size)
    if n_reference <= n_neighbors:
        raise ValueError(
            f"Only {n_available} chain points carry nonzero weight, leaving "
            f"{n_reference} for the reference sample after a pool of {pool_size}; "
            f"at least {n_neighbors + 1} are needed. Lengthen the chain "
            f"(sampling.n_steps) or lower acquisition.pool_factor."
        )
    draw = np.random.choice(
        a=len(chain_unique), size=pool_size + n_reference, replace=False, p=weights
    )
    pool_indices, reference_indices = draw[:pool_size], draw[pool_size:]
    pool = chain_unique[pool_indices]
    reference = chain_unique[reference_indices]

    # Calculate the whitened coordinates for later distance computations
    pool_whitened = knn_index.whiten(pool)

    # The k-NN estimator does not report the density *at* a point: it reports the
    # density averaged over a ball enclosing a fixed mass fraction k/N, and in high
    # dimensions that ball spans much of the distribution (r_k ≈ 0.9 √n at n = 30).
    # The estimate is therefore flattened, log ρ̂ ≈ s log ρ + c with s < 1 (s ≈ 0.64 at
    # n = 30, k = 20). Comparing an *analytic* target density against an *estimated*
    # current density would let that exponent through and drive the training set to
    # L^(1/(sT)) rather than L^(1/T) -- systematically colder than requested.
    # Estimating both sides with the same estimator, the same k and the same whitening
    # makes it cancel: log D̂ = s [log ρ_target - log ρ_merged] vanishes exactly where
    # the true ratio is one, whatever s happens to be. A wrong s then only compresses
    # the size of the deficit signal, i.e. the rate of convergence, not its target.
    log_ball_vol = _log_unit_ball_volume(ndim)
    reference_index = _WhitenedKNN(
        reference, n_neighbors, transform=knn_index.transform
    )
    reference_distances, _ = reference_index.query(pool)
    r_k_reference = np.max(reference_distances, axis=1)

    # ρ_ref = k / (V_n * r_{k,ref}^n), the number density of the n_reference-point
    # reference cloud. The selection loop rescales it to the point count of the merged
    # training set, which grows as points are appended.
    log_rho_reference = (
        np.log(n_neighbors) - log_ball_vol - ndim * np.log(r_k_reference)
    )

    # Compute the k-th nearest neighbor distances for the pool points
    pool_distances, _ = knn_index.query(pool)
    current_distances = pool_distances.copy()

    selected_indices = []
    selected_mask = np.ones(pool_size, dtype=bool)
    pool_rows = np.arange(pool_size)

    for _ in range(min(n_append, pool_size)):
        # Compute the k-th nearest neighbor distances for the merged set of points (current training data + selected points)
        r_k_merged = np.max(current_distances, axis=1)
        log_rho_merged = np.log(n_neighbors) - log_ball_vol - ndim * np.log(r_k_merged)

        # Compute the next-point target number density by rescaling the reference
        # cloud's density from n_reference points to n_current + 1.
        #
        # This rescale is deliberately the naive one, and it is not what an unbiased
        # estimator would need: because the estimate is flattened, the estimator's
        # response to scaling a density by λ is s log λ, not log λ, so a full log factor
        # overshoots by (1 - s) log[(n_current + 1) / n_reference]. That residual is
        # independent of θ, so it only shifts how eagerly points are added, never where
        # -- log D keeps the form (θ-independent constant) + s [log q - log ρ], whose
        # θ-dependence still vanishes exactly at ρ ∝ q for any s. It is zero at the
        # start of the batch (n_reference == n_current) and grows to (1 - s) log(1 +
        # n_append / n_current) by the end: 0.01 to 0.07 nats for batch fractions
        # between 4% and 20%.
        log_rho_target = np.log(n_current + 1) - np.log(n_reference) + log_rho_reference

        # Compute the number density ratio D = ρ_target / ρ_merged
        log_D = log_rho_target - log_rho_merged

        # Compute the positive number deficit retention factor
        # a = [1 - 1/D]_+ = [1 - e^{-logD}]_+
        a = np.zeros_like(log_D)
        positive = log_D > 0.0
        a[positive] = -np.expm1(-log_D[positive])
        a[~selected_mask] = 0.0

        # Normalize the retention factors and sample without replacement
        a_sum = np.sum(a)
        if not np.isfinite(a_sum) or a_sum <= 0.0:
            break
        a = a / a_sum
        next_index = np.random.choice(
            a=pool_size,
            p=a,
        )
        selected_indices.append(next_index)
        selected_mask[next_index] = False
        n_current += 1

        # Distance from every pool point to the newly selected point
        new_distance = np.sqrt(
            ((pool_whitened - pool_whitened[next_index]) ** 2).sum(axis=1)
        )

        # Keep the k smallest distances from each pool point to
        # the current training data plus all selected points
        worst_indices = np.argmax(current_distances, axis=1)
        worst_distances = current_distances[pool_rows, worst_indices]
        improved = new_distance < worst_distances
        current_distances[pool_rows[improved], worst_indices[improved]] = new_distance[
            improved
        ]

        # The selected point itself should not be treated as a future query candidate
        current_distances[next_index] = np.inf
    metrics = {
        "n_unique": len(chain_unique),
        "max_multiplicity": int(np.max(counts)),
        "n_reference": n_reference,
    }

    return pool[selected_indices], metrics
