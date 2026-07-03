import best
import numpy as np
import tensorflow as tf

class BestInferenceSampler:
    """
    Wrapper around best.Sampler (https://github.com/AndreasNygaard/best-inference).

    Supports methods: 'mh', 'aies', 'hmc', 'nuts', 'mala'.

    Runs n_chains independent replicas. For 'aies' each replica is an ensemble of
    n_walkers interacting walkers; for the other methods each replica is a single
    independent chain and n_walkers is not used. The independent-replica structure
    is handled natively inside best.Sampler.

    Parameters
    ----------
    n_walkers : int
        Ensemble size per replica for 'aies' (must be even). Ignored otherwise.
    n_chains : int
        Number of independent replicas (ensembles / chains).
    ndim : int
        Number of parameters.
    logpost_fn : callable
        Accepts a tf.Tensor of shape (n, ndim) and returns a tf.Tensor of
        shape (n,) with log-posterior values.
    method : str
        Sampling algorithm passed to best.Sampler.sample().
    bounds : tuple, optional
        (lower_bounds, upper_bounds) where each is a list of length ndim.
        Enables boundary enforcement and sets a sensible initial covariance.
    """
    def __init__(self, n_walkers, n_chains, ndim, logpost_fn, method, bounds=None):
        self.n_walkers  = n_walkers
        self.n_chains   = n_chains
        self.ndim       = ndim
        self.method     = method
        if bounds is not None:
            bounds = tuple(np.asarray(bound, dtype=np.float32) for bound in bounds)
        self._sampler   = best.Sampler(logpost_fn, bounds=bounds, enforce_boundaries=False)
        self._chain     = None
        self._log_prob  = None
        self._n_steps   = 0
        self._accept    = 0.0

    def run(self, initial_pos, max_steps):
        # initial_pos: (n_chains, n_walkers, ndim). 'aies' uses the full ensemble
        # structure; the other methods take one walker per replica (n independent
        # chains), since n_walkers is meaningful only for 'aies'.
        initial_pos = np.asarray(initial_pos, dtype=np.float32)
        if self.method == 'aies':
            initial_state = initial_pos
            n_walkers = self.n_walkers
        else:
            initial_state = initial_pos[:, 0, :]
            n_walkers = None
        results = self._sampler.sample(
            initial_state=tf.convert_to_tensor(initial_state, dtype=tf.float32),
            method=self.method,
            n_steps=max_steps,
            n_chains=self.n_chains,
            n_walkers=n_walkers,
            num_burnin_steps=0,
            num_covmat_updates=0,
        )
        # aies:  samples (n_steps, n_chains, n_walkers, ndim)
        # other: samples (n_steps, n_chains, ndim)
        self._chain    = np.asarray(results.samples, dtype=np.float32)
        self._log_prob = np.asarray(results.log_prob, dtype=np.float32)
        self._n_steps  = max_steps
        self._accept   = float(np.asarray(results.acceptance_rate))

    def get_chain(self, flat=False, discard=0, thin=1):
        chain = self._chain[discard::thin]
        if flat:
            chain = chain.reshape(-1, self.ndim)
        return chain

    def get_logpost(self, flat=False, discard=0, thin=1):
        logp = self._log_prob[discard::thin]
        if flat:
            logp = logp.reshape(-1)
        return logp

    def free_memory(self):
        self._chain    = None
        self._log_prob = None

    def get_acceptance_fraction(self):
        return self._accept

    def get_n_steps(self):
        return self._n_steps

def build_sampler(name, n_walkers, n_chains, ndim, logpost_fn, bounds=None):
    return BestInferenceSampler(
        n_walkers=n_walkers, 
        n_chains=n_chains, 
        ndim=ndim,
        logpost_fn=logpost_fn, 
        method=name, 
        bounds=bounds
    )
