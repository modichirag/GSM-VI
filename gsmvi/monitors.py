import numpy as np
import os
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, grad, random
from numpyro.distributions import MultivariateNormal

from . import plotting

# def reverse_kl(samples, lpq, lpp):
#     logl = jnp.mean(lpp(samples))
#     logq = jnp.mean(lpq(samples))
#     rkl = logq - logl
#     return rkl

# def forward_kl(samples, lpq, lpp):
#     logl = jnp.mean(lpp(samples))
#     logq = jnp.mean(lpq(samples))
#     fkl = logl - logq
#     return fkl

@partial(jit, static_argnums=(3))
def reverse_kl(samples, mu, cov, lp):
    q = MultivariateNormal(mu, cov)
    logq = jnp.sum(q.log_prob(samples))
    logl = jnp.sum(lp(samples))
    rkl = logq - logl
    rkl /= samples.shape[0]
    return rkl

@partial(jit, static_argnums=(3))
def forward_kl(samples, mu, cov, lp):
    q = MultivariateNormal(mu, cov)
    logq = jnp.sum(q.log_prob(samples))
    logl = jnp.sum(lp(samples))
    fkl = logl - logq
    fkl /= samples.shape[0]
    return fkl

    
@dataclass
class KLMonitor():
    """
    Class to monitor KL divergence during optimization for VI
    
    Inputs:
    
    batch_size: (int) Number of samples to use to estimate divergence
    checkpoint: (int) Number of iterations after which to run monitor
    savepoint : (int) Number of iterations after which to save progress
    offset_evals: (int) Value with which to offset number of gradient evaluatoins
                    Used to account for gradient evaluations done in warmup or initilization        
    ref_samples: Optional, samples from the target distribution.
                   If provided, also track forward KL divergence
    savepath : Optional, directory to save the losses and plots at saavepoints.
               If None, no plots are saved.
    plot_samples : Optional, bool. If True, plot loss function at savepoints. 
      
    """
    
    batch_size : int = 8
    checkpoint : int = 10
    savepoint : int = 100
    offset_evals : int = 0
    savepath : str = None
    ref_samples : np.array = None
    plot_samples : bool = False
    
    def __post_init__(self):

        self.rkl = []
        self.fkl = []
        self.nevals = []        
        
    def reset(self,
              batch_size=None, 
              checkpoint=None,
              savepoint=None,
              offset_evals=None,
              ref_samples=None,
              plot_samples=None,
              savepath=None):
        self.nevals = []
        self.rkl = []
        self.fkl = []
        if batch_size is not None: self.batch_size = batch_size
        if checkpoint is not None: self.checkpoint = checkpoint
        if savepoint is not None: self.savepoint = savepoint
        if offset_evals is not None: self.offset_evals = offset_evals
        if ref_samples is not None: self.ref_samples = ref_samples
        if plot_samples is not None: self.plot_samples = plot_samples
        if savepath is not None: self.savepath = savepath
        print('offset evals reset to : ', self.offset_evals)
        
    def __call__(self, i, params, lp, key, nevals=1):
        """
        Main function to monitor reverse (and forward) KL divergence over iterations.
        If savepath is not None, it also saves and plots losses at savepoints.
        
        Inputs:
        
        i: (int) iteration number
        params: (tuple; (mean, cov)) Current estimate of mean and covariance matrix
        savepoint : (int) Number of iterations after which to save progress
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
            qsamples = np.random.multivariate_normal(mean=mu, cov=cov, size=self.batch_size)
            #q = MultivariateNormal(loc=mu, covariance_matrix=cov)
            #q_lp = jit(lambda x: jnp.sum(q.log_prob(x)))
            self.rkl.append(reverse_kl(qsamples, mu, cov, lp))

            if self.ref_samples is not None:
                idx = np.random.permutation(self.ref_samples.shape[0])[:self.batch_size]
                psamples = self.ref_samples[idx]
                self.fkl.append(forward_kl(psamples, mu,cov, lp))
        except Exception as e:
            print(f"Exception occured in monitor : {e}.\nAppending NaN")
            self.rkl.append(np.NaN)
            self.fkl.append(np.NaN)
            
        self.nevals.append(self.offset_evals + nevals)
        self.offset_evals = self.nevals[-1]

        #save
        if (self.savepath is not None) & (i%self.savepoint == 0):

            print("Savepoint: saving current fit, loss and diagnostic plots")
            
            os.makedirs(self.savepath, exist_ok=True)
            np.save(f"{self.savepath}/mu_fit", mu)
            np.save(f"{self.savepath}/cov_fit", cov)
            np.save(f"{self.savepath}/nevals", self.nevals)
            np.save(f"{self.savepath}/rkl", self.rkl)
            plotting.plot_loss(self.nevals, self.rkl, self.savepath, fname='rkl', logit=True)
            if self.ref_samples is not None:
                np.save(f"{self.savepath}/fkl", self.fkl)
                plotting.plot_loss(self.nevals, self.fkl, self.savepath, fname='fkl', logit=True)

            if self.plot_samples:
                try:
                    qsamples = np.random.multivariate_normal(mean=mu, cov=cov, size=1000)
                    plotting.corner(qsamples[:500],
                                    savepath=f"{self.savepath}/",
                                    savename=f"corner{i}", maxdims=5) 

                    plotting.compare_hist(qsamples, ref_samples=self.ref_samples[:1000],
                                    savepath=f"{self.savepath}/",
                                    savename=f"hist{i}") 
                except Exception as e:
                    print(f"Exception occured in plotting samples in monitor : {e}.\nSkip")
            
        return key
    
