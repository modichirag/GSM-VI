import numpy as np
import os
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import jit, grad, random
from numpyro.distributions import MultivariateNormal

import plotting

def backward_kl(samples, lpq, lpp):
    logl = jnp.sum(lpp(samples))
    logq = jnp.sum(lpq(samples))
    bkl = logq - logl
    return bkl

def forward_kl(samples, lpq, lpp):
    logl = jnp.sum(lpp(samples))
    logq = jnp.sum(lpq(samples))
    fkl = logl - logq
    return fkl

    
@dataclass
class KLMonitor():

    batch_size : int = 8
    checkpoint : int = 10
    savepoint : int = 100
    offset_evals : int = 0
    savepath : str = None
    ref_samples : np.array = None
    plot_samples : bool = False
    
    def __post_init__(self):

        self.bkl = []
        self.fkl = []
        self.nevals = []        
        
    def reset(self):
        self.offset_evals = 0
        self.nevals = []
        self.bkl = []
        self.fkl = []

        
    def __call__(self, i, params, lp, key, nevals=1):

        #
        mu, cov = params
        key, key_sample = random.split(key)
        qsamples = np.random.multivariate_normal(mean=mu, cov=cov, size=self.batch_size)
        q = MultivariateNormal(loc=mu, covariance_matrix=cov)
        self.bkl.append(backward_kl(qsamples, q.log_prob, lp))

        if self.ref_samples is not None:
            idx = np.random.permutation(self.ref_samples.shape[0])[:self.batch_size]
            psamples = self.ref_samples[idx]
            self.fkl.append(forward_kl(psamples, q.log_prob, lp))
            
        self.nevals.append(self.offset_evals + nevals)
        self.offset_evals = self.nevals[-1]

        #save
        if (self.savepath is not None) & (i%self.savepoint == 0):
            os.makedirs(self.savepath, exist_ok=True)
            np.save(f"{self.savepath}/nevals", self.nevals)
            np.save(f"{self.savepath}/bkl", self.bkl)
            plotting.plot_loss(self.nevals, self.bkl, self.savepath, fname='bkl', logit=True)
            if self.ref_samples is not None:
                np.save(f"{self.savepath}/fkl", self.fkl)
                plotting.plot_loss(self.nevals, self.fkl, self.savepath, fname='fkl', logit=True)

            if self.plot_samples:
                qsamples = np.random.multivariate_normal(mean=mu, cov=cov, size=1000)
                plotting.corner(qsamples,
                                savepath=f"{self.savepath}/",
                                savename=f"corner{i}") 

                plotting.compare_hist(qsamples, ref_samples=self.ref_samples[:1000],
                                savepath=f"{self.savepath}/",
                                savename=f"hist{i}") 
            
        return key
    
