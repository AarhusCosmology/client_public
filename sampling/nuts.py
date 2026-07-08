import tensorflow as tf
import tensorflow_probability as tfp

from .base import BaseSampler


class NUTSampler(BaseSampler):
    """
    No-U-Turn Sampler (Hoffman & Gelman 2014) via tfp.mcmc.NoUTurnSampler
    """

    def __init__(self, n_chains, ndim, log_prob_fn, initial_step_size=0.5, max_tree_depth=5):
        self.n_chains = n_chains
        self.ndim = ndim
        self.log_prob_fn = log_prob_fn
        self.initial_step_size = initial_step_size
        self.max_tree_depth = max_tree_depth

        self._kernel = tfp.mcmc.NoUTurnSampler(
            target_log_prob_fn=log_prob_fn,
            step_size=self.initial_step_size,
            max_tree_depth=self.max_tree_depth,
        )

        self._pos = None
        self._chain = None
        self._log_prob = None
        self._accept_count = None
        self._n_proposals = 0

    def _initialize(self, initial_positions):
        self._pos = tf.cast(tf.convert_to_tensor(initial_positions), tf.float32)
        self._chain = None
        self._log_prob = None
        self._accept_count = None
        self._n_proposals = 0

    @tf.function(jit_compile=True, reduce_retracing=True)
    def _run_graph(self, n_steps, burn_in, pos):
        chain, (is_accepted, log_prob) = tfp.mcmc.sample_chain(
            num_results=n_steps,
            num_burnin_steps=burn_in,
            current_state=pos,
            kernel=self._kernel,
            trace_fn=lambda _, pkr: (pkr.is_accepted, pkr.target_log_prob),
        )
        accept_count = tf.reduce_sum(tf.cast(is_accepted, tf.int32), axis=0)
        return chain[-1], chain, log_prob, accept_count

    def run(self, n_steps, initial_positions=None, burn_in=0, progress=True):
        if initial_positions is not None:
            self._initialize(initial_positions)
        if self._pos is None:
            raise ValueError("Sampler not initialized. Provide initial_positions.")

        if not progress:
            self._pos, self._chain, self._log_prob, self._accept_count = self._run_graph(
                tf.convert_to_tensor(n_steps, dtype=tf.int32), 
                tf.convert_to_tensor(burn_in, dtype=tf.int32),
                self._pos
            )
            self._n_proposals = n_steps
            return

        from tqdm.auto import tqdm

        pos = self._pos
        chain_chunks, log_prob_chunks = [], []
        total_accept_count = tf.zeros((self.n_chains,), dtype=tf.int32)

        remaining_burn_in = burn_in

        with tqdm(total=burn_in + n_steps, unit="step") as pbar:
            for start in range(0, n_steps, 100):
                chunk_size = min(100, n_steps - start)
                pos, chain, log_prob, accept_count = self._run_graph(
                    tf.convert_to_tensor(chunk_size, dtype=tf.int32),
                    tf.convert_to_tensor(remaining_burn_in, dtype=tf.int32), 
                    pos
                )
                chain_chunks.append(chain)
                log_prob_chunks.append(log_prob)
                total_accept_count += accept_count
                pbar.update(remaining_burn_in + chunk_size)

                remaining_burn_in = 0

        self._pos = pos
        self._chain = tf.concat(chain_chunks, axis=0)
        self._log_prob = tf.concat(log_prob_chunks, axis=0)
        self._accept_count = total_accept_count
        self._n_proposals = n_steps

    def chain(self, discard=0, thin=1):
        return self._chain[discard::thin]

    def log_prob(self, discard=0, thin=1):
        return self._log_prob[discard::thin]

    def acceptance_fraction(self):
        return tf.cast(self._accept_count, tf.float32) / tf.cast(
            self._n_proposals, tf.float32
        )

    def reset(self):
        self._pos = None
        self._chain = None
        self._log_prob = None
        self._accept_count = None
        self._n_proposals = 0
