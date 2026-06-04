import tensorflow as tf


def gumbel_select(chain, logposts, n, sampling_temperature=1.0, target_temperature=1.0):
    # Weighted resampling without replacement using the Gumbel-max trick:
    # adding i.i.d. Gumbel noise to log-weights and taking the top-k indices
    # gives the exact joint distribution of weighted sampling without replacement.
    # tf.cast is a no-op for the TF EnsembleSampler and handles the numpy
    # arrays returned by emceeSampler, keeping the rest of the pipeline uniform.
    # IS correction: chain ~ L^{1/T} pi, target ~ L^{1/T'} pi, so w ∝ L^{1/T' - 1/T}.
    # Since logposts = (1/T) log L, log w = (T/T' - 1) * logposts.
    chain = tf.cast(chain, tf.float32)
    logposts = tf.cast(logposts, tf.float32)
    log_weights = (sampling_temperature / target_temperature - 1.0) * logposts
    log_weights -= tf.reduce_max(log_weights)
    gumbel = -tf.math.log(-tf.math.log(
        tf.random.uniform(tf.shape(log_weights), dtype=tf.float32)
    ))
    _, indices = tf.math.top_k(log_weights + gumbel, k=n)
    return tf.gather(chain, indices).numpy()
