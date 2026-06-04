# sampling/sampler.py

from abc import ABC, abstractmethod
import time
import emcee
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from sampling.autocorr import integrated_autocorr_time


class BaseSampler(ABC):
    @abstractmethod
    def run(self, initial_pos, max_steps, **kwargs):
        pass

    @abstractmethod
    def get_chain(self, **kwargs):
        pass

    @abstractmethod
    def get_logpost(self, **kwargs):
        pass


class EmceeSampler(BaseSampler):
    """Wrapper around emcee.EnsembleSampler using a TF-compiled logpost_fn."""

    def __init__(self, n_walkers, ndim, logpost_fn, vectorize=True):
        self.n_walkers = n_walkers
        self.ndim = ndim

        @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32)])
        def _compiled(positions):
            return logpost_fn(positions)

        def _wrapped(positions):
            return _compiled(tf.cast(positions, tf.float32)).numpy()

        self._sampler = emcee.EnsembleSampler(
            nwalkers=n_walkers,
            ndim=ndim,
            log_prob_fn=_wrapped,
            vectorize=vectorize,
        )

    def run(self, initial_pos, max_steps, chunk_size=None, target_ess=None,
            tau_stability=0.01, iat_memory_mb=None, progress=True):
        if chunk_size is None or target_ess is None:
            self._sampler.run_mcmc(initial_pos, max_steps, progress=progress)
            return

        prev_tau = None
        steps_done = 0
        pos = initial_pos
        pbar = tqdm(total=max_steps, desc="Sampling") if progress else None
        log = pbar.write if pbar is not None else print

        while steps_done < max_steps:
            steps_this_chunk = min(chunk_size, max_steps - steps_done)
            self._sampler.run_mcmc(pos, steps_this_chunk, progress=False)
            pos = None
            steps_done += steps_this_chunk
            if pbar is not None:
                pbar.update(steps_this_chunk)

            if iat_memory_mb is not None:
                max_elements = int(iat_memory_mb * 1024 * 1024 / 4)
                thin = max(1, (steps_done * self.n_walkers * self.ndim) // max_elements)
            else:
                thin = 1
            try:
                tau_np = np.asarray(self._sampler.get_autocorr_time(thin=thin, quiet=False))
                reliable = True
            except emcee.autocorr.AutocorrError:
                tau_np = np.asarray(self._sampler.get_autocorr_time(thin=thin, quiet=True))
                reliable = False

            ess = self.n_walkers * steps_done / tau_np

            if not reliable:
                log(f"  [{steps_done}/{max_steps}] IAT unreliable — "
                    f"max(tau)={tau_np.max():.1f}, min(ESS)={ess.min():.0f} "
                    f"(need N > {50 * tau_np.max():.0f} for reliable estimate)")
                continue

            ess_ok = bool(np.all(ess > target_ess))
            stable = True
            if tau_stability is not None and prev_tau is not None:
                stable = bool(np.all(np.abs(tau_np - prev_tau) / prev_tau < tau_stability))

            log(f"  [{steps_done}/{max_steps}] "
                f"min(ESS)={ess.min():.0f}/{target_ess}, "
                f"max(tau)={tau_np.max():.1f}, "
                f"tau_stable={stable}")

            prev_tau = tau_np
            if ess_ok and stable:
                log(f"  ESS convergence reached at step {steps_done}.")
                break

        if pbar is not None:
            pbar.close()

    def get_chain(self, flat=False, discard=0, thin=1):
        return self._sampler.get_chain(flat=flat, discard=discard, thin=thin)

    def get_logpost(self, flat=False, discard=0, thin=1):
        return self._sampler.get_log_prob(flat=flat, discard=discard, thin=thin)


class EnsembleSampler(BaseSampler):
    """Affine-invariant ensemble sampler (Goodman & Weare 2010) in TensorFlow.

    Parameters
    ----------
    n_walkers : int
        Number of walkers. Must be even and >= 2 * ndim.
    ndim : int
        Number of parameters.
    logpost_fn : callable
        Accepts a tf.Tensor of shape (n_walkers, ndim) and returns a
        tf.Tensor of shape (n_walkers,) with log-posterior values.
    a : float
        Stretch-move scale parameter (default 2.0).
    """

    def __init__(self, n_walkers, ndim, logpost_fn, a=2.0):
        if n_walkers % 2 != 0:
            raise ValueError("n_walkers must be even")
        self.n_walkers = n_walkers
        self.ndim = ndim
        self.logpost_fn = logpost_fn
        self._half = n_walkers // 2
        self._ndim_m1 = tf.constant(ndim - 1, dtype=tf.float32)

        sqrt_a = float(a ** 0.5)
        self._z_offset = tf.constant(1.0 / sqrt_a,          dtype=tf.float32)
        self._z_range  = tf.constant(sqrt_a - 1.0 / sqrt_a, dtype=tf.float32)

        self._chain    = None
        self._log_prob = None
        self._compiled_run = tf.function(self._run_steps, jit_compile=True)

    def _step(self, pos, logp):
        half = self._half

        u0         = tf.random.uniform((half,), dtype=tf.float32)
        z0         = (u0 * self._z_range + self._z_offset) ** 2
        j0         = tf.random.uniform((half,), minval=half, maxval=self.n_walkers, dtype=tf.int32)
        x_partner0 = tf.gather(pos, j0)
        pos0       = pos[:half]
        proposals0 = x_partner0 + tf.reshape(z0, (-1, 1)) * (pos0 - x_partner0)
        prop_logp0 = self.logpost_fn(proposals0)
        log_acc0   = self._ndim_m1 * tf.math.log(z0) + prop_logp0 - logp[:half]
        accepted0  = tf.math.log(tf.random.uniform((half,), dtype=tf.float32)) < log_acc0
        new_pos0   = tf.where(tf.reshape(accepted0, (-1, 1)), proposals0, pos0)
        new_logp0  = tf.where(accepted0, prop_logp0, logp[:half])

        u1         = tf.random.uniform((half,), dtype=tf.float32)
        z1         = (u1 * self._z_range + self._z_offset) ** 2
        j1         = tf.random.uniform((half,), minval=0, maxval=half, dtype=tf.int32)
        x_partner1 = tf.gather(new_pos0, j1)
        pos1       = pos[half:]
        proposals1 = x_partner1 + tf.reshape(z1, (-1, 1)) * (pos1 - x_partner1)
        prop_logp1 = self.logpost_fn(proposals1)
        log_acc1   = self._ndim_m1 * tf.math.log(z1) + prop_logp1 - logp[half:]
        accepted1  = tf.math.log(tf.random.uniform((half,), dtype=tf.float32)) < log_acc1
        new_pos1   = tf.where(tf.reshape(accepted1, (-1, 1)), proposals1, pos1)
        new_logp1  = tf.where(accepted1, prop_logp1, logp[half:])

        return tf.concat([new_pos0, new_pos1], axis=0), tf.concat([new_logp0, new_logp1], axis=0)

    def _run_steps(self, pos, logp, n_steps):
        chain_ta = tf.TensorArray(dtype=tf.float32, size=n_steps, dynamic_size=False)
        logp_ta  = tf.TensorArray(dtype=tf.float32, size=n_steps, dynamic_size=False)

        def body(i, pos, logp, chain_ta, logp_ta):
            pos, logp = self._step(pos, logp)
            return i + 1, pos, logp, chain_ta.write(i, pos), logp_ta.write(i, logp)

        _, _, _, chain_ta, logp_ta = tf.while_loop(
            lambda i, *_: i < n_steps,
            body,
            loop_vars=[0, pos, logp, chain_ta, logp_ta],
            parallel_iterations=1,
        )
        return chain_ta.stack(), logp_ta.stack()

    def iat(self, chain=None, c=5.0, iat_memory_mb=None):
        return integrated_autocorr_time(self._chain if chain is None else chain, c, iat_memory_mb)

    def run(self, initial_pos, max_steps, chunk_size=None, target_ess=None,
            tau_stability=0.01, iat_memory_mb=None, progress=True):
        pos  = tf.cast(initial_pos, tf.float32)
        logp = self.logpost_fn(pos)

        _display_chunk = min(max(1, max_steps // 100), 1000)
        _dummy_pos  = tf.zeros((self.n_walkers, self.ndim), dtype=tf.float32)
        _dummy_logp = tf.zeros((self.n_walkers,),            dtype=tf.float32)
        self._compiled_run(_dummy_pos, _dummy_logp, tf.constant(_display_chunk))
        _remainder = max_steps % _display_chunk
        if _remainder > 0:
            self._compiled_run(_dummy_pos, _dummy_logp, tf.constant(_remainder))

        if chunk_size is None or target_ess is None:
            chain_buf = tf.Variable(tf.zeros((max_steps, self.n_walkers, self.ndim), dtype=tf.float32), trainable=False)
            logp_buf  = tf.Variable(tf.zeros((max_steps, self.n_walkers),            dtype=tf.float32), trainable=False)
            steps_done = 0
            pbar = tqdm(total=max_steps, desc="Sampling") if progress else None
            while steps_done < max_steps:
                steps_this = min(_display_chunk, max_steps - steps_done)
                chunk_chain, chunk_logp = self._compiled_run(pos, logp, tf.constant(steps_this))
                pos  = chunk_chain[-1]
                logp = chunk_logp[-1]
                chain_buf[steps_done:steps_done + steps_this].assign(chunk_chain)
                logp_buf[steps_done:steps_done + steps_this].assign(chunk_logp)
                steps_done += steps_this
                if pbar is not None:
                    pbar.update(steps_this)
            if pbar is not None:
                pbar.close()
            self._chain    = chain_buf
            self._log_prob = logp_buf
            return

        # Chunk mode with ESS-based early stopping
        if chunk_size is not None:
            self._compiled_run(_dummy_pos, _dummy_logp, tf.constant(chunk_size))
        _remainder2 = max_steps % chunk_size if chunk_size else 0
        if _remainder2 > 0:
            self._compiled_run(_dummy_pos, _dummy_logp, tf.constant(_remainder2))

        chain_buf = tf.Variable(tf.zeros((max_steps, self.n_walkers, self.ndim), dtype=tf.float32), trainable=False)
        logp_buf  = tf.Variable(tf.zeros((max_steps, self.n_walkers),            dtype=tf.float32), trainable=False)
        prev_tau   = None
        steps_done = 0
        pbar = tqdm(total=max_steps, desc="Sampling") if progress else None

        while steps_done < max_steps:
            steps_this_chunk = min(chunk_size, max_steps - steps_done)
            chunk_chain, chunk_logp = self._compiled_run(pos, logp, tf.constant(steps_this_chunk))
            pos  = chunk_chain[-1]
            logp = chunk_logp[-1]
            start = steps_done
            steps_done += steps_this_chunk
            chain_buf[start:steps_done].assign(chunk_chain)
            logp_buf[start:steps_done].assign(chunk_logp)
            if pbar is not None:
                pbar.update(steps_this_chunk)

            full_chain    = chain_buf[:steps_done]
            tau, reliable = self.iat(full_chain, iat_memory_mb=iat_memory_mb)
            log = pbar.write if pbar is not None else print
            tau_np = tau.numpy()
            ess    = self.n_walkers * steps_done / tau_np

            if not reliable.numpy():
                log(f"  [{steps_done}/{max_steps}] IAT unreliable — "
                    f"max(tau)={tau_np.max():.1f}, min(ESS)={ess.min():.0f} "
                    f"(need N > {50 * tau_np.max():.0f} for reliable estimate)")
                continue

            ess_ok = bool(np.all(ess > target_ess))
            stable = True
            if tau_stability is not None and prev_tau is not None:
                stable = bool(np.all(np.abs(tau_np - prev_tau) / prev_tau < tau_stability))

            log(f"  [{steps_done}/{max_steps}] "
                f"min(ESS)={ess.min():.0f}/{target_ess}, "
                f"max(tau)={tau_np.max():.1f}, "
                f"tau_stable={stable}")

            prev_tau = tau_np
            if ess_ok and stable:
                log(f"  ESS convergence reached at step {steps_done}.")
                break

        if pbar is not None:
            pbar.close()
        self._chain    = chain_buf[:steps_done]
        self._log_prob = logp_buf[:steps_done]

    def get_chain(self, flat=False, discard=0, thin=1):
        chain = self._chain[discard::thin]
        if flat:
            chain = tf.reshape(chain, (-1, self.ndim))
        return chain

    def get_logpost(self, flat=False, discard=0, thin=1):
        logp = self._log_prob[discard::thin]
        if flat:
            logp = tf.reshape(logp, (-1,))
        return logp


def build_sampler(name, n_walkers, ndim, logpost_fn):
    if name == 'ensemble':
        return EnsembleSampler(n_walkers=n_walkers, ndim=ndim, logpost_fn=logpost_fn)
    if name == 'emcee':
        return EmceeSampler(n_walkers=n_walkers, ndim=ndim, logpost_fn=logpost_fn)
    raise ValueError(f"Unknown sampler: '{name}'. Available: ['ensemble', 'emcee']")

