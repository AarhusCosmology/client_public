import tensorflow as tf


class AffineInvariantEnsembleSampler:
    """
    Affine-invariant ensemble sampler (Goodman & Weare 2010) in TensorFlow
    """
    def __init__(self, n_walkers, ndim, log_prob_fn, a=2.0):
        if n_walkers % 2 != 0:
            raise ValueError("Number of walkers must be even.")
        self.n_walkers = n_walkers
        self.ndim = ndim
        self.log_prob_fn = log_prob_fn
        self.a = a

        # Precompute constants used at every step to avoid repeated Python/Tensor allocations.
        self._half = n_walkers // 2
        self._ndim_m1 = tf.constant(ndim - 1, dtype=tf.float32)
        sqrt_a = float(a ** 0.5)

        # Stretch-move support mapped from u ~ U(0, 1): z in [1/a, a].
        self._z_offset = tf.constant(1.0 / sqrt_a, dtype=tf.float32)
        self._z_range = tf.constant(sqrt_a - 1.0 / sqrt_a, dtype=tf.float32)
        self._pos = None
        self._logp = None
        self._chain = None
        self._log_prob = None
        self._accept_count = None
        self._n_proposals = 0

    def _initialize(self, initial_positions):
        self._pos = tf.cast(tf.convert_to_tensor(initial_positions), tf.float32)
        self._logp = self.log_prob_fn(self._pos)
        self._chain = None
        self._log_prob = None
        self._accept_count = None
        self._n_proposals = 0

    def _update_half(self, active_positions, active_log_probs, partner_pool):
        # z ~ g(z) ∝ 1/√z on [1/a, a]
        # Inverse CDF sampling gives
        # z = (u * (√a - 1/√a) + 1/√a)², where u ~ U(0, 1)
        u = tf.random.uniform((self._half,), dtype=tf.float32)
        z = tf.square(u * self._z_range + self._z_offset)
        partner_indices = tf.random.uniform((self._half,), maxval=self._half, dtype=tf.int32)
        partner_positions = tf.gather(partner_pool, partner_indices)
        proposals = partner_positions + tf.expand_dims(z, axis=1) * (active_positions - partner_positions)
        proposal_log_probs = self.log_prob_fn(proposals)

        # r = z^(ndim-1) * proposal_prob / active_prob
        log_accept_ratio = self._ndim_m1 * tf.math.log(z) + proposal_log_probs - active_log_probs

        # a = min(1, r)
        accepted = tf.math.log(tf.random.uniform((self._half,), dtype=tf.float32)) < log_accept_ratio
        new_positions = tf.where(tf.expand_dims(accepted, axis=1), proposals, active_positions)
        new_log_probs = tf.where(accepted, proposal_log_probs, active_log_probs)

        return new_positions, new_log_probs, accepted

    def _step(self, pos, logp):
        # Split-ensemble updates: each walker proposes against a complementary partner set.
        # Updating halves sequentially is the standard Goodman-Weare scheme.
        new_positions, new_log_probs, accepted = self._update_half(
            active_positions=pos[:self._half],
            active_log_probs=logp[:self._half],
            partner_pool=pos[self._half:],
        )

        # The second half conditions on the newly updated first half.
        new_positions2, new_log_probs2, accepted2 = self._update_half(
            active_positions=pos[self._half:],
            active_log_probs=logp[self._half:],
            partner_pool=new_positions,
        )

        return (
            tf.concat([new_positions, new_positions2], axis=0),
            tf.concat([new_log_probs, new_log_probs2], axis=0),
            tf.concat([accepted, accepted2], axis=0),
        )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def _run_graph(self, n_steps, pos, logp):
        # TensorArray avoids Python-side appends and keeps storage on the TF side.
        chain = tf.TensorArray(
            dtype=tf.float32,
            size=n_steps,
            element_shape=tf.TensorShape([self.n_walkers, self.ndim])
        )
        log_prob = tf.TensorArray(
            dtype=tf.float32,
            size=n_steps,
            element_shape=tf.TensorShape([self.n_walkers])
        )

        accept_count = tf.zeros((self.n_walkers,), dtype=tf.int32)

        def cond(i, pos, logp, chain, log_prob, accept_count):
            return i < n_steps

        def body(i, pos, logp, chain, log_prob, accept_count):
            pos, logp, accepted = self._step(pos, logp)
            chain = chain.write(i, pos)
            log_prob = log_prob.write(i, logp)
            accept_count += tf.cast(accepted, tf.int32)
            return i + 1, pos, logp, chain, log_prob, accept_count

        _, pos, logp, chain, log_prob, accept_count = tf.while_loop(
            cond, body,
            loop_vars=[tf.constant(0, dtype=tf.int32), pos, logp, chain, log_prob, accept_count],
            # One logical MCMC iteration per loop body; keep execution order explicit.
            parallel_iterations=1,
        )

        return pos, logp, chain.stack(), log_prob.stack(), accept_count

    def run(self, n_steps, initial_positions=None, progress=True):
        if initial_positions is not None:
            self._initialize(initial_positions)
        if self._pos is None or self._logp is None:
            raise ValueError("Sampler not initialized. Provide initial_positions.")

        if not progress:
            self._pos, self._logp, self._chain, self._log_prob, self._accept_count = self._run_graph(
                tf.convert_to_tensor(n_steps, dtype=tf.int32),
                self._pos,
                self._logp,
            )
            self._n_proposals = n_steps
            return

        from tqdm.auto import tqdm

        pos, logp = self._pos, self._logp
        chain_chunks, log_prob_chunks = [], []
        total_accept_count = tf.zeros((self.n_walkers,), dtype=tf.int32)
        with tqdm(total=n_steps, unit="step") as pbar:
            for start in range(0, n_steps, 1000):
                chunk_size = min(1000, n_steps - start)
                pos, logp, chain, log_prob, accept_count = self._run_graph(
                    tf.convert_to_tensor(chunk_size, dtype=tf.int32),
                    pos,
                    logp,
                )
                chain_chunks.append(chain)
                log_prob_chunks.append(log_prob)
                total_accept_count += accept_count
                pbar.update(chunk_size)
        self._pos = pos
        self._logp = logp
        self._chain = tf.concat(chain_chunks, axis=0)
        self._log_prob = tf.concat(log_prob_chunks, axis=0)
        self._accept_count = total_accept_count
        self._n_proposals = n_steps

    def chain(self, discard=0, thin=1):
        return self._chain[discard::thin]

    def log_prob(self, discard=0, thin=1):
        return self._log_prob[discard::thin]

    def acceptance_fraction(self):
        return tf.cast(self._accept_count, tf.float32) / tf.cast(self._n_proposals, tf.float32)

    def reset(self):
        self._pos = None
        self._logp = None
        self._chain = None
        self._log_prob = None
        self._accept_count = None
        self._n_proposals = 0


def build_sampler(name, n_walkers, ndim, log_prob_fn):
    if name == "aies":
        return AffineInvariantEnsembleSampler(
            n_walkers=n_walkers,
            ndim=ndim,
            log_prob_fn=log_prob_fn
        )
    raise ValueError(f"Unknown sampler name: {name}. Available samplers: ['aies']")
