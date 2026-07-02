import numpy as np
import tensorflow as tf

from scipy.special import gammaln, logsumexp

def _log_unit_ball_volume(n_dims):
    return (n_dims / 2.0) * np.log(np.pi) - gammaln(n_dims / 2.0 + 1.0)

def select_points(dataset, chain, logposts, surrogate, n_append, mcmc_temperature, pool_factor):
    knn_index = dataset.knn_index
    n_dims = knn_index.n_dims
    n_neighbors = dataset.n_neighbors
    target_temperature = dataset.target_temperature
    n_current = len(dataset.inputs)

    # log w_i ∝ (1/T - 1/T_MC) * loglkls_i
    # Since logposts = 1/T_MC * loglkls, we have
    # log w_i ∝ (T_MC/T - 1) * logposts_i
    log_weights_raw = (mcmc_temperature / target_temperature - 1.0) * logposts

    # Collapse exact duplicate MCMC states, but preserve their total probability mass
    chain_unique, first_idx, inverse, counts = np.unique(
        chain,
        axis=0,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )
    logposts_unique = logposts[first_idx]
    log_weights_unique = log_weights_raw[first_idx] + np.log(counts)

    # Calculate and print the deduplication statistics
    n_duplicates = len(chain) - len(chain_unique)
    unique_fraction = len(chain_unique) / len(chain)
    max_multiplicity = np.max(counts)
    print(
        f"Surrogate chain deduplication: "
        f"{len(chain)} samples -> {len(chain_unique)} unique "
        f"({n_duplicates} duplicates, unique_fraction={unique_fraction:.3f}, "
        f"max_multiplicity={max_multiplicity})"
    )

    # Normalize the unique-point weights
    log_weights_unique = log_weights_unique - logsumexp(log_weights_unique)
    weights = np.exp(log_weights_unique)

    # Weighted sample without replacement
    pool_size = min(pool_factor * n_append, len(chain_unique))
    pool_indices = np.random.choice(
        a=len(chain_unique),
        size=pool_size,
        replace=False,
        p=weights
    )
    pool = chain_unique[pool_indices]

    # Calculate the whitened coordinates for later distance computations
    pool_whitened = knn_index.whiten(pool)

    # log Z_surr = logsumexp_i ( 1/T * loglkl_surr_i - log ρ_i ),
    # with ρ_i = (k - 1) / (V_n * r_{k,i}^n), using a leave-one-out estimator, where r_{k,i} is the
    # distance from the i-th training point to its k-th nearest other training point
    training_data_distances, _ = knn_index.query(dataset.inputs, n_neighbors + 1)
    r_k_train = training_data_distances[:, n_neighbors]
    log_ball_vol = _log_unit_ball_volume(n_dims)
    log_rho_train = np.log(n_neighbors - 1) - log_ball_vol - n_dims * np.log(r_k_train)
    loglkl_surr_train = surrogate.loglkl(tf.constant(dataset.inputs)).numpy()
    log_Z_surr = logsumexp(loglkl_surr_train / target_temperature - log_rho_train)

    # Compute the surrogate target probability density on the pool points
    loglkls_surr_pool = logposts_unique[pool_indices] * mcmc_temperature
    log_q_surr = loglkls_surr_pool / target_temperature - log_Z_surr

    # Compute the k-th nearest neighbor distances for the pool points
    pool_distances, _ = knn_index.query(pool)
    selected_distances = np.full((pool_size, n_neighbors), np.inf)

    selected_indices = []
    selected_mask = np.ones(pool_size, dtype=bool)

    for _ in range(min(n_append, pool_size)):
        # Compute the k-th nearest neighbor distances for the merged set of points (current training data + selected points)
        merged_distances = np.concatenate([pool_distances, selected_distances], axis=1)
        r_k_merged = np.partition(merged_distances, n_neighbors - 1, axis=1)[:, n_neighbors - 1]
        log_rho_merged = np.log(n_neighbors) - log_ball_vol - n_dims * np.log(r_k_merged)

        # Compute the next-point target number density
        # log ρ_target = log[ (n_current + 1) * q_surr ]
        log_rho_target = np.log(n_current + 1) + log_q_surr

        # Compute the number density ratio D = ρ_target / ρ_merged
        # log D = log[ (n_current + 1) * q_surr ] - log ρ_merged
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
            print("Density deficit exhausted: no under-represented candidates remain")
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
        new_distance = np.sqrt(((pool_whitened - pool_whitened[next_index]) ** 2).sum(axis=1, keepdims=True))

        # Add the new selected point's distance to the existing selected distances
        combined = np.concatenate([selected_distances, new_distance], axis=1)

        # Keep the k smallest distances for each pool point
        selected_distances = np.partition(combined, n_neighbors - 1, axis=1)[:, :n_neighbors]

        # The selected point itself should not be treated as a future query candidate
        selected_distances[next_index] = np.inf

    return pool[selected_indices]
