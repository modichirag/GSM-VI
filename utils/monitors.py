import numpy as np
import os
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import jit, grad, random
from numpyro.distributions import MultivariateNormal

import plotting

def backward_kl(samples, lpq, lpp):
    logl = np.mean(lpp(samples))
    logq = np.mean(lpq(samples))
    bkl = logq - logl
    return bkl

def forward_kl(samples, lpq, lpp):
    logl = np.mean(lpp(samples))
    logq = np.mean(lpq(samples))
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
        
    def reset(self,
              batch_size=None, 
              checkpoint=None,
              savepoint=None,
              offset_evals=None,
              ref_samples=None,
              plot_samples=None,
              savepath=None):
        self.nevals = []
        self.bkl = []
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

        #
        mu, cov = params
        key, key_sample = random.split(key)
        np.random.seed(key_sample[0])

        try:
            qsamples = np.random.multivariate_normal(mean=mu, cov=cov, size=self.batch_size)
            q = MultivariateNormal(loc=mu, covariance_matrix=cov)
            self.bkl.append(backward_kl(qsamples, q.log_prob, lp))

            if self.ref_samples is not None:
                idx = np.random.permutation(self.ref_samples.shape[0])[:self.batch_size]
                psamples = self.ref_samples[idx]
                self.fkl.append(forward_kl(psamples, q.log_prob, lp))
        except Exception as e:
            print(f"Exception occured in monitor : {e}.\nAppending NaN")
            self.bkl.append(np.NaN)
            self.fkl.append(np.NaN)
            
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
                plotting.corner(qsamples[:500],
                                savepath=f"{self.savepath}/",
                                savename=f"corner{i}", maxdims=5) 

                plotting.compare_hist(qsamples, ref_samples=self.ref_samples[:1000],
                                savepath=f"{self.savepath}/",
                                savename=f"hist{i}") 
            
        return key
    
