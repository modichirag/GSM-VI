import numpy as np
from dataclasses import dataclass
from functools import partial

import jax.numpy as jnp
from jax import jit, grad, random
from numpyro.distributions import MultivariateNormal


def reverse_kl(samples, lpq, lpp):
    logl = np.sum(lpp(samples))
    logq = np.sum(lpq(samples))
    rkl = logq - logl
    rkl /= samples.shape[0]
    return rkl

def forward_kl(samples, lpq, lpp):
    logl = np.sum(lpp(samples))
    logq = np.sum(lpq(samples))
    fkl = logl - logq
    fkl /= samples.shape[0]
    return fkl

@partial(jit, static_argnums=(3))
def reverse_kl_jit(samples, mu, cov, lp):
    q = MultivariateNormal(mu, cov)
    logq = jnp.sum(q.log_prob(samples))
    logl = jnp.sum(lp(samples))
    rkl = logq - logl
    rkl /= samples.shape[0]
    return rkl

@partial(jit, static_argnums=(3))
def forward_kl_jit(samples, mu, cov, lp):
    q = MultivariateNormal(mu, cov)
    logq = jnp.sum(q.log_prob(samples))
    logl = jnp.sum(lp(samples))
    fkl = logl- logq
    fkl /= samples.shape[0]
    return fkl

    
@dataclass
class KLMonitor():
    """
    Class to monitor KL divergence during optimization for VI
    
    Inputs:
    
    batch_size_kl: (int) Number of samples to use to estimate KL divergence
    checkpoint: (int) Number of iterations after which to run monitor
    offset_evals: (int) Value with which to offset number of gradient evaluatoins
                    Used to account for gradient evaluations done in warmup or initilization        
    ref_samples: Optional, samples from the target distribution.
                   If provided, also track forward KL divergence      
    """
    
    batch_size_kl : int = 8
    checkpoint : int = 20
    offset_evals : int = 0
    ref_samples : np.array = None
    
    def __post_init__(self):

        self.rkl = []
        self.fkl = []
        self.nevals = []        
        
    def reset(self,
              batch_size_kl=None, 
              checkpoint=None,
              offset_evals=None,
              ref_samples=None ):
        self.nevals = []
        self.rkl = []
        self.fkl = []
        if batch_size_kl is not None: self.batch_size_kl = batch_size_kl
        if checkpoint is not None: self.checkpoint = checkpoint
        if offset_evals is not None: self.offset_evals = offset_evals
        if ref_samples is not None: self.ref_samples = ref_samples
        print('offset evals reset to : ', self.offset_evals)
        
    def __call__(self, i, params, lp, key, nevals=1):
        """
        Main function to monitor reverse (and forward) KL divergence over iterations.

        Inputs:
        
        i: (int) iteration number
        params: (tuple; (mean, cov)) Current estimate of mean and covariance matrix
        lp: Function to evaluate target log-probability
        key: Random number generator key (jax.random.PRNGKey)
        nevals: (int) Number of gradient evaluations SINCE the last call of the monitor function

        Returns:
        key : New key for generation random number
        """

        #
        mu, cov = params
        key, key_sample = random.split(key)
        np.random.seed(key_sample[0])

        
        try:
            qsamples = np.random.multivariate_normal(mean=mu, cov=cov, size=self.batch_size_kl)
            q = MultivariateNormal(loc=mu, covariance_matrix=cov)
            self.rkl.append(reverse_kl(qsamples, q.log_prob, lp))

            if self.ref_samples is not None:
                idx = np.random.permutation(self.ref_samples.shape[0])[:self.batch_size_kl]
                psamples = self.ref_samples[idx]
                self.fkl.append(forward_kl(psamples, q.log_prob, lp))
            else:
                self.fkl.append(np.NaN)
            
        except Exception as e:
            print(f"Exception occured in monitor : {e}.\nAppending NaN")
            self.rkl.append(np.NaN)
            self.fkl.append(np.NaN)
            
        self.nevals.append(self.offset_evals + nevals)
        self.offset_evals = self.nevals[-1]

        return key
    
