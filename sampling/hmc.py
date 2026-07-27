import tensorflow as tf
import tensorflow_probability as tfp

from .base import BaseSampler


class HMCSampler(BaseSampler):
    """
    Hamiltonian Monte Carlo (Duane et al. 1987) via tfp.mcmc.HamiltonianMonteCarlo

    Uses the same transformation strategy as NUTSampler: when prior bounds
    are provided, leapfrog integration runs in the unconstrained space of a
    Sigmoid bijector so the -inf logprior walls are never encountered, and
    when a covariance matrix is provided a whitening bijector is composed
    inside the Sigmoid so the target has unit covariance there, acting as a
    dense mass matrix.

    Unlike NUTS, the trajectory length is fixed (10 leapfrog steps) rather
    than chosen per-step by a U-turn criterion; only the step size is adapted
    (dual averaging during burn-in, targeting the 0.65 acceptance rate that
    is optimal for HMC).
    """

    def __init__(self, n_chains, ndim, log_prob_fn, covmat=None, bounds=None):
        self.n_chains = n_chains
        self.ndim = ndim
        self.log_prob_fn = log_prob_fn

        if bounds is not None:
            self._lower = tf.cast(tf.convert_to_tensor(bounds[0]), tf.float32)
            self._upper = tf.cast(tf.convert_to_tensor(bounds[1]), tf.float32)
            bijectors = [tfp.bijectors.Sigmoid(low=self._lower, high=self._upper)]
            # d(unconstrained)/d(constrained) for the sigmoid, evaluated at the
            # midpoint of the bounds where it has the closed form 4 / (u - l).
            # The posterior sits deep inside the bounds, where the sigmoid is
            # near-affine, so this linearization is accurate over its support.
            jacobian = 4.0 / (self._upper - self._lower)
        else:
            self._lower = None
            self._upper = None
            bijectors = []
            jacobian = None

        if covmat is None:
            # No geometry available: fall back to an isotropic step size,
            # re-expressed in unconstrained units when a sigmoid is in play.
            initial_step_size = tf.fill((ndim,), tf.constant(0.5, dtype=tf.float32))
            if jacobian is not None:
                initial_step_size = initial_step_size * jacobian
        else:
            covmat = tf.cast(tf.convert_to_tensor(covmat), tf.float32)
            if jacobian is not None:
                covmat = jacobian[:, None] * covmat * jacobian[None, :]
            # Ridge the covariance so the Cholesky stays well defined for chains
            # with near-degenerate directions. The floor is relative to the mean
            # marginal variance and sized for float32, whose eps is ~1e-7.
            ridge = 1e-6 * tf.linalg.trace(covmat) / ndim
            covmat = covmat + ridge * tf.eye(ndim, dtype=tf.float32)
            bijectors.append(
                tfp.bijectors.ScaleMatvecTriL(tf.linalg.cholesky(covmat))
            )
            # The whitened space has unit covariance by construction.
            initial_step_size = tf.constant(1.0, dtype=tf.float32)

        self.bijector = (
            tfp.bijectors.Chain(bijectors) if bijectors else tfp.bijectors.Identity()
        )
        self.initial_step_size = initial_step_size

        self._pos = None
        self._chain = None
        self._log_prob = None
        self._accept_count = None
        self._n_proposals = 0

    def _initialize(self, initial_positions):
        pos = tf.cast(tf.convert_to_tensor(initial_positions), tf.float32)
        if self._lower is not None:
            # Positions exactly on a bound map to +-inf in unconstrained space.
            margin = 1e-6 * (self._upper - self._lower)
            pos = tf.clip_by_value(pos, self._lower + margin, self._upper - margin)
        self._pos = pos
        self._chain = None
        self._log_prob = None
        self._accept_count = None
        self._n_proposals = 0

    @tf.function(jit_compile=True, reduce_retracing=True)
    def _run_graph(self, n_steps, burn_in, pos, previous_kernel_results):
        inner_kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=self.log_prob_fn,
            step_size=self.initial_step_size,
            num_leapfrog_steps=10,
        )

        transformed_kernel = tfp.mcmc.TransformedTransitionKernel(
            inner_kernel=inner_kernel,
            bijector=self.bijector,
        )

        adaptive_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=transformed_kernel,
            num_adaptation_steps=burn_in,
            target_accept_prob=0.65,
        )

        chain, is_accepted, kernel_results = tfp.mcmc.sample_chain(
            num_results=n_steps,
            num_burnin_steps=0 if previous_kernel_results is not None else burn_in,
            current_state=pos,
            kernel=adaptive_kernel,
            previous_kernel_results=previous_kernel_results,
            trace_fn=lambda _, pkr: pkr.inner_results.inner_results.is_accepted,
            return_final_kernel_results=True,
        )
        if self._lower is not None:
            # float32 rounding in the sigmoid can place states an ulp outside
            # the bounds, where the -inf logprior would poison the log-probs.
            chain = tf.clip_by_value(chain, self._lower, self._upper)
        # Recompute the log-posterior on the constrained states rather than
        # undoing the bijector's Jacobian from the unconstrained-space value,
        # which is not numerically stable for states on the bounds.
        log_prob = tf.reshape(
            self.log_prob_fn(tf.reshape(chain, (-1, self.ndim))),
            tf.shape(chain)[:2],
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
            for start in range(0, n_steps, 1000):
                chunk_size = min(1000, n_steps - start)
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
