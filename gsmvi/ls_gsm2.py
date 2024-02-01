import jax
import jax.numpy as jnp
from jax import jit, random
from jax.scipy.linalg import sqrtm
#from scipy.linalg import sqrtm
#from numpyro.distributions import MultivariateNormal  ##Needed if sampling from numpyro dist below
import numpy as np              
from scipy.optimize import minimize
from numpyro.distributions import MultivariateNormal
from functools import partial
#from timeout import timeout
import timeout_decorator
import os

@partial(jit, static_argnums=(1))
def inner_loop(reg, lp, mu0, S0, xbar, C, gbar, G, gout, muout, isamples, lps):
    print("jit")
    reg = jnp.exp(reg)
    U = reg * G + (reg)/(1+reg) * gout #jnp.outer(gbar, gbar)
    V = S0 + reg * C + (reg)/(1+reg) * muout #jnp.outer(mu0 - xbar, mu0 - xbar)
    I = jnp.identity(U.shape[1])

    mat = I + 4 * jnp.matmul(U, V)
    #S = 2 * jnp.matmul(V, jnp.linalg.inv(I + sqrtm(mat).real))
    S = 2 * jnp.linalg.solve(I + sqrtm(mat).real.T, V.T)
    mu = 1/(1+reg) * mu0 + reg/(1+reg) * (jnp.matmul(S, gbar) + xbar)
    jitter = 1e-6
    S += jnp.eye(mu.size)*jitter # jitter covariance matrix
    S = (S  + S.T)/2.
    q = MultivariateNormal(loc=mu, covariance_matrix=S)

    # key = jax.random.PRNGKey(0)
    # isamples = q.sample(key, (isamples.shape[1],))
    # logl = jnp.sum(lp(isamples))
    # logq = jnp.sum(q.log_prob(isamples))
    # elbo = logl - logq

    elbo = jnp.sum(lps * (lps - q.log_prob(isamples)))
    # elbo = jnp.sum(q.log_prob(isamples) * (q.log_prob(isamples)- lps))

    negelbo = -1. * elbo
    return negelbo



#@timeout(100, os.strerror(errno.ETIMEDOUT))
@timeout_decorator.timeout(5)
def get_reg(reg, lp, mu0, S0, xbar, C, gbar, G, samples, lps):
    gout = jnp.outer(gbar, gbar)
    muout = jnp.outer(mu0 - xbar, mu0 - xbar)
    res = minimize(inner_loop, jnp.array([jnp.log(reg)]),
                   args=(lp, mu0, S0, xbar, C, gbar, G, gout, muout, samples, lps),
                   method='Nelder-Mead', #options={"maxiter":samples.shape[0], 'atol':1e-3, 'rtol':1e-3},
                   )
    print(reg, jnp.exp(res.x), res.nfev, res.message, res.status)
    #if res.status != 0 : raise
    return res
    
#@jit
def ls_gsm_update(lp, samples, vs, mu0, S0, reg):
    """
    Returns updated mean and covariance matrix with GSM updates.
    For a batch, this is simply the mean of updates for individual samples.

    Inputs:
      samples: Array of samples of shape BxD where B is the batch dimension
      vs : Array of score functions of shape BxD corresponding to samples
      mu0 : Array of shape D, current estimate of the mean
      S0 : Array of shape DxD, current estimate of the covariance matrix

    Returns:
      mu : Array of shape D, new estimate of the mean
      S : Array of shape DxD, new estimate of the covariance matrix
    """

    assert len(samples.shape) == 2
    assert len(vs.shape) == 2
    B = samples.shape[0]
    xbar = jnp.mean(samples, axis=0)
    outer_map = jax.vmap(jnp.outer, in_axes=(0, 0))
    xdiff = samples - xbar
    C = jnp.mean(outer_map(xdiff, xdiff), axis=0)

    gbar = jnp.mean(vs, axis=0)
    gdiff = vs - gbar
    G = jnp.mean(outer_map(gdiff, gdiff), axis=0) 
    
    lps = lp(samples)
    key = jax.random.PRNGKey(0)

    try:
        res = get_reg(reg, lp, mu0, S0, xbar, C, gbar, G, samples, lps)
        reg = (jnp.exp(res.x) * reg )**0.5
    except Exception as e:
        print(e)

    U = reg * G + (reg)/(1+reg) * jnp.outer(gbar, gbar)
    V = S0 + reg * C + (reg)/(1+reg) * jnp.outer(mu0 - xbar, mu0 - xbar)
    I = jnp.identity(samples.shape[1])
    
    mat = I + 4 * jnp.matmul(U, V)
    # S = 2 * jnp.matmul(V, jnp.linalg.inv(I + sqrtm(mat).real))
    S = 2 * jnp.linalg.solve(I + sqrtm(mat).real.T, V.T)
    mu = 1/(1+reg) * mu0 + reg/(1+reg) * (jnp.matmul(S, gbar) + xbar)
    
    return mu, S, reg



class LS_GSM:
    """
    Wrapper class for using GSM updates to fit a distribution
    """
    def __init__(self, D, lp, lp_g):
        """
        Inputs:
          D: (int) Dimensionality (number) of parameters
          lp : Function to evaluate target log-probability distribution. 
               (Only used in monitor, not for fitting)
          lp_g : Function to evaluate score, i.e. the gradient of the target log-probability distribution
        """
        self.D = D
        self.lp = lp
        self.lp_g = lp_g

        
    def fit(self, key, regf, mean=None, cov=None, batch_size=2, niter=5000, nprint=10, verbose=True, check_goodness=True, monitor=None, retries=10, jitter=1e-6):
        """
        Main function to fit a multivariate Gaussian distribution to the target

        Inputs:
          key: Random number generator key (jax.random.PRNGKey)
          mean : Function to return regularizer value at an iteration. See Regularizers class below
          mean : Optional, initial value of the mean. Expected None or array of size D
          cov : Optional, initial value of the covariance matrix. Expected None or array of size DxD
          batch_size : Optional, int. Number of samples to match scores for at every iteration
          niter : Optional, int. Total number of iterations 
          nprint : Optional, int. Number of iterations after which to print logs
          verbose : Optional, bool. If true, print number of iterations after nprint
          check_goodness : Optional, bool. Recommended. Wether to check floating point errors in covariance matrix update
          monitor : Optional. Function to monitor the progress and track different statistics for diagnostics. 
                    Function call should take the input tuple (iteration number, [mean, cov], lp, key, number of grad evals).
                    Example of monitor class is provided in utils/monitors.py

        Returns:
          mu : Array of shape D, fit of the mean
          cov : Array of shape DxD, fit of the covariance matrix
        """
        if mean is None:
            mean = jnp.zeros(self.D)
        if cov is None:
            cov = jnp.identity(self.D)

        nevals = 1
        
        if nprint > niter: nprint = niter    
        for i in range(niter + 1):
            if (i%(niter//nprint) == 0) and verbose : 
                print(f'Iteration {i} of {niter}')
                
            if monitor is not None:
                if (i%monitor.checkpoint) == 0:
                    monitor(i, [mean, cov], self.lp, key, nevals=nevals)
                    nevals = 0
                    
            # Can generate samples from jax distribution (commented below), but using numpy is faster
            j = 0
            while True:         # Sometimes run crashes due to a bad sample. Avoid that by re-trying. 
                try:
                    key, key_sample = random.split(key, 2) 
                    np.random.seed(key_sample[0])
                    samples = np.random.multivariate_normal(mean=mean, cov=cov, size=batch_size)
                    # samples = MultivariateNormal(loc=mean, covariance_matrix=cov).sample(key, (batch_size,))
                    vs = self.lp_g(samples)
                    nevals += batch_size
                    if i == 0 : reg = regf(i)
                    reg = max(reg, 1e-5)
                    #reg = regf(i)
                    mean_new, cov_new, reg = ls_gsm_update(self.lp, samples, vs, mean, cov, reg)
                    cov_new += np.eye(self.D)*jitter # jitter covariance matrix
                    cov_new = (cov_new + cov_new.T)/2.
                    break
                except Exception as e:
                    if j < retries :
                        j += 1 
                        print(f"Failed with exception {e}")
                        print(f"Trying again {j} of {retries}")
                    else : raise e
            
            is_good = self._check_goodness(cov_new)
            if is_good:
                mean, cov = mean_new, cov_new
            else:
                if verbose: print("Bad update for covariance matrix. Revert")
                    
        if monitor is not None:
            monitor(i, [mean, cov], self.lp, key, nevals=nevals)                    
        return mean, cov

    
    def _check_goodness(self, cov):
        '''
        Internal function to check if the new covariance matrix is a valid covariance matrix. 
        Required due to floating point errors in updating the convariance matrix directly, 
        insteead of it's Cholesky form. 
        '''
        is_good = False
        try:
            if (np.isnan(np.linalg.cholesky(cov))).any():
                #nan_update.append(j)
                pass
            else:
                is_good = True
            return is_good
        except:
            return is_good



class Regularizers():
    """
    Class for regularizers used in LS_GSM
    """

    def __init__(self):

        self.counter = 0

    def reset(self):

        self.counter = 0

        
    def constant(self, reg0):

        def reg_iter(iteration):
            self.counter +=1 
            return reg0
        return reg_iter

    
    def linear(self, reg0):

        def reg_iter(iteration):
            self.counter += 1
            return reg0/self.counter
        
        return reg_iter

    
    def custom(self, func):
        
        def reg_iter(iteration):
            self.counter += 1
            return func(self.counter)
        
        return reg_iter
    