import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad, random
from numpyro.distributions import MultivariateNormal
import optax
from functools import partial      


class NGD():
    """
    Class for fitting a multivariate Gaussian distribution with dense covariance matrix
    by following natural gradients
    """
    
    def __init__(self, D, lp, lp_g):
        """
        Inputs:
          D: (int) Dimensionality (number) of parameters.
          lp : Function to evaluate target log-probability distribution
               whose gradient can be evaluated with jax.grad(lp)
        """
        self.D = D
        self.lp = lp
        self.lp_g = lp_g
   

    @partial(jit, static_argnums=(0, 5, 6))
    def ngd_update(self, samples, vs, mu, cov, lr=1e-1, reg=1e-5):

        icov = jnp.linalg.inv(cov)
        #
        g_p = vs
        g_q = -jnp.dot(icov, (samples-mu).T).T
        g = jnp.mean(g_q - g_p, axis=0)

        h_p = jax.vmap(lambda gg: jnp.outer(gg, gg), in_axes=(0))(g_p).mean(axis=0)
        h_q = -icov
        h = h_q + h_p + jnp.eye(self.D)*reg

        icovnew = icov + lr * h    
        covnew = jnp.linalg.inv(icovnew)
        munew = mu - lr * jnp.dot(covnew, g)
        return munew, covnew


    
    def fit(self, key, mean=None, cov=None, lr=1e-2, reg=1e-5, batch_size=8, niter=1000, nprint=10, monitor=None, verbose=True, check_goodness=True):
        """
        Main function to fit a multivariate Gaussian distribution to the target

        Inputs:
          key: Random number generator key (jax.random.PRNGKey)
          mean : Optional, initial value of the mean. Expected None or array of size D
          cov : Optional, initial value of the covariance matrix. Expected None or array of size DxD
          batch_size : Optional, int. Number of samples to match scores for at every iteration
          niter : Optional, int. Total number of iterations 
          nprint : Optional, int. Number of iterations after which to print logs
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
            
        for i in range(niter + 1):
            if (i%(niter//nprint) == 0) and verbose : 
                print(f'Iteration {i} of {niter}')
                
            if monitor is not None:
                if (i%monitor.checkpoint) == 0:
                    monitor(i, [mean, cov], self.lp, key, nevals=nevals)
                    nevals = 0
                    
            # Can generate samples from jax distribution (commented below), but using numpy is faster
            key, key_sample = random.split(key, 2) 
            np.random.seed(key_sample[0])
            samples = np.random.multivariate_normal(mean=mean, cov=cov, size=batch_size)
            vs = self.lp_g(samples)
            mean_new, cov_new = self.ngd_update(samples, vs, mean, cov, lr=lr, reg=reg)
            nevals += batch_size
            
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
                nan_update.append(j)
            else:
                is_good = True
            return is_good
        except:
            return is_good
