import tensorflow as tf


def _next_pow_two(n: int) -> int:
    """Smallest power of 2 that is >= n."""
    p = 1
    while p < n:
        p <<= 1
    return p


def _autocorr_1d_batch(x: tf.Tensor) -> tf.Tensor:
    """Normalised ACF for a batch of 1D chains via FFT.

    Zero-pads to the next power of 2 >= 2*N before transforming to
    eliminate circular aliasing and keep the FFT fast.

    Parameters
    ----------
    x : tf.Tensor, shape (..., n_steps)
        Zero-mean chains; leading dimensions are treated as a batch.

    Returns
    -------
    tf.Tensor, shape (..., n_steps)
        Normalised ACF with acf[..., 0] == 1.
    """
    n_int = int(tf.shape(x)[-1])
    n_fft = _next_pow_two(2 * n_int)

    # Autocorrelation via the convolution theorem: IFFT(|FFT(x)|^2).
    # rfft pads the input to n_fft; irfft without fft_length infers n_fft.
    f       = tf.signal.rfft(x, fft_length=[n_fft])
    acf_raw = tf.signal.irfft(f * tf.math.conj(f))[..., :n_int]

    # Normalise so acf[..., 0] == 1.
    return acf_raw / acf_raw[..., :1]


def integrated_autocorr_time(chain: tf.Tensor, c: float = 5.0, iat_memory_mb: float = None):
    """Estimate the integrated autocorrelation time (IAT) per parameter.

    Uses the Foreman-Mackey (2017) method: the normalised ACF is computed
    independently for each walker via FFT, averaged across walkers to
    reduce estimator variance, and the Sokal (1989) automated window is
    applied to select the truncation lag M.

    The window M is the smallest index satisfying M >= c * tau(M), where
    tau(M) = 1 + 2 * sum_{t=1}^{M} rho(t). Choosing c ~ 5 balances
    truncation bias against high-lag noise variance.

    Parameters
    ----------
    chain : tf.Tensor, shape (n_steps, n_walkers, ndim)
    c : float
        Sokal window constant. Default 5.0.
    iat_memory_mb : float, optional
        Memory budget in MB for the IAT FFT computation. A thinning factor
        thin = max(1, (n_steps * n_walkers * ndim) // max_elements) is
        computed so the chain tensor fed into the FFT never exceeds this
        many megabytes (float32, 4 bytes per element). This bounds VRAM
        usage regardless of chain length, walker count, or parameter count.
        The returned tau is corrected back to original-step units via
        tau_original = thin * tau_thinned. Default None (use the full chain).

    Returns
    -------
    tau : tf.Tensor, shape (ndim,)
        Estimated IAT per parameter, in units of original (un-thinned) steps.
    reliable : tf.Tensor, scalar bool
        True when n_steps > 50 * max(tau). Below this threshold the
        windowed estimate is likely biased and should not be trusted.
    """
    chain = tf.cast(chain, tf.float32)  # (n_steps, n_walkers, ndim)
    n_steps_full = int(tf.shape(chain)[0])
    if iat_memory_mb is not None:
        n_walkers_i  = int(tf.shape(chain)[1])
        ndim_i       = int(tf.shape(chain)[2])
        max_elements = int(iat_memory_mb * 1024 * 1024 / 4)
        thin = max(1, (n_steps_full * n_walkers_i * ndim_i) // max_elements)
    else:
        thin = 1
    if thin > 1:
        chain = chain[::thin]

    # Rearrange to (n_walkers, ndim, n_steps) for a single batched ACF call.
    x = tf.transpose(chain, perm=[1, 2, 0])

    # Subtract per-(walker, parameter) mean so the ACF is well-defined.
    x = x - tf.reduce_mean(x, axis=-1, keepdims=True)

    # ACF for every (walker, parameter) pair: (n_walkers, ndim, n_steps).
    acf_per_walker = _autocorr_1d_batch(x)

    # Average ACFs across walkers — reduces estimator variance compared
    # to computing the ACF of the walker-mean chain (Goodman & Weare 2010).
    mean_acf = tf.reduce_mean(acf_per_walker, axis=0)  # (ndim, n_steps)

    # Running IAT estimate: tau(M) = 1 + 2 * cumsum(rho)[M] - 1
    taus = 2.0 * tf.cumsum(mean_acf, axis=-1) - 1.0  # (ndim, n_steps)

    # --- Sokal automated window ---
    # Find the smallest M where M >= c * tau(M), i.e. the first M where
    # the "not yet converged" condition M < c * tau(M) flips to False.
    n_steps = tf.shape(taus)[-1]
    ndim    = tf.shape(taus)[0]
    m_idx   = tf.cast(tf.range(n_steps), tf.float32)  # (n_steps,)

    not_converged = m_idx[tf.newaxis, :] < c * taus  # (ndim, n_steps)

    # argmin on int(bool) returns the first 0 (first False).
    # If all True (window never found), argmin returns 0 — use n_steps-1 instead.
    window_found = tf.reduce_any(~not_converged, axis=-1)
    first_false  = tf.cast(
        tf.argmin(tf.cast(not_converged, tf.int32), axis=-1), tf.int32
    )
    window = tf.where(window_found, first_false, tf.fill([ndim], n_steps - 1))

    # Gather tau[d, window[d]] for each parameter d.
    indices = tf.stack([tf.range(ndim), window], axis=-1)  # (ndim, 2)
    tau     = tf.gather_nd(taus, indices)                   # (ndim,)

    # Reliability heuristic: the windowed estimate needs N > 50 * max(tau).
    reliable = tf.cast(n_steps, tf.float32) > 50.0 * tf.reduce_max(tau)

    return tau * tf.cast(thin, tf.float32), reliable
