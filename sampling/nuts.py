import tensorflow as tf
import tensorflow_probability as tfp

from .base import BaseSampler


class NUTSampler(BaseSampler):
    """
    No-U-Turn Sampler (Hoffman & Gelman 2014) via tfp.mcmc.NoUTurnSampler
    """

    def __init__(self, n_chains, ndim, log_prob_fn, initial_step_size=0.5):
        self.n_chains = n_chains
        self.ndim = ndim
        self.log_prob_fn = log_prob_fn
        self.initial_step_size = initial_step_size

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
    def _run_graph(self, n_steps, burn_in, pos, previous_kernel_results):
        inner_kernel = tfp.mcmc.NoUTurnSampler(
            target_log_prob_fn=self.log_prob_fn,
            step_size=self.initial_step_size,
        )

        adaptive_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=inner_kernel,
            num_adaptation_steps=burn_in,
            target_accept_prob=0.75,
        )

        chain, (is_accepted, log_prob), kernel_results = tfp.mcmc.sample_chain(
            num_results=n_steps,
            num_burnin_steps=0 if previous_kernel_results is not None else burn_in,
            current_state=pos,
            kernel=adaptive_kernel,
            previous_kernel_results=previous_kernel_results,
            trace_fn=lambda _, pkr: (pkr.inner_results.is_accepted, pkr.inner_results.target_log_prob),
            return_final_kernel_results=True,
        )
        accept_count = tf.reduce_sum(tf.cast(is_accepted, tf.int32), axis=0)
        return chain[-1], chain, log_prob, accept_count, kernel_results

    def run(self, n_steps, initial_positions=None, burn_in=0, progress=True):
        if initial_positions is not None:
            self._initialize(initial_positions)
        if self._pos is None:
            raise ValueError("Sampler not initialized. Provide initial_positions.")

        if not progress:
            self._pos, self._chain, self._log_prob, self._accept_count, _ = self._run_graph(
                tf.convert_to_tensor(n_steps, dtype=tf.int32),
                tf.convert_to_tensor(burn_in, dtype=tf.int32),
                self._pos,
                None,
            )
            self._n_proposals = n_steps
            return

        from tqdm.auto import tqdm

        pos = self._pos
        chain_chunks, log_prob_chunks = [], []
        total_accept_count = tf.zeros((self.n_chains,), dtype=tf.int32)
        kernel_results = None

        with tqdm(total=burn_in + n_steps, unit="step") as pbar:
            for start in range(0, n_steps, 100):
                chunk_size = min(100, n_steps - start)
                pos, chain, log_prob, accept_count, kernel_results = self._run_graph(
                    tf.convert_to_tensor(chunk_size, dtype=tf.int32),
                    tf.convert_to_tensor(burn_in, dtype=tf.int32),
                    pos,
                    kernel_results,
                )
                chain_chunks.append(chain)
                log_prob_chunks.append(log_prob)
                total_accept_count += accept_count
                pbar.update((burn_in if start == 0 else 0) + chunk_size)

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
