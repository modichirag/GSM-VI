import jax
import jax.numpy as jnp
from jax import jit, random
#from numpyro.distributions import MultivariateNormal  ##Needed if sampling from numpyro dist below
import numpy as np              

@jit
def _gsm_update_single(sample, v, mu0, S0):
    '''returns GSM update to mean and covariance matrix for a single sample
    '''
    S0v = jnp.matmul(S0, v)
    vSv = jnp.matmul(v, S0v)
    mu_v = jnp.matmul((mu0 - sample), v)
    rho = 0.5 * jnp.sqrt(1 + 4*(vSv + mu_v**2)) - 0.5
    eps0 = S0v - mu0 + sample

    #mu update
    mu_vT = jnp.outer((mu0 - sample), v)
    den = 1 + rho + mu_v
    I = jnp.eye(sample.shape[0])
    mu_update = 1/(1 + rho) * jnp.matmul(( I - mu_vT / den), eps0)
    mu = mu0 + mu_update

    #S update
    Supdate_0 =  jnp.outer((mu0-sample), (mu0-sample))
    Supdate_1 =  jnp.outer((mu-sample), (mu-sample))
    S_update = (Supdate_0 - Supdate_1)
    return mu_update, S_update


@jit
def gsm_update(samples, vs, mu0, S0):
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

    vgsm_update = jax.vmap(_gsm_update_single, in_axes=(0, 0, None, None))
    mu_update, S_update = vgsm_update(samples, vs, mu0, S0)
    mu_update = jnp.mean(mu_update, axis=0)
    S_update = jnp.mean(S_update, axis=0)
    mu = mu0 + mu_update
    S = S0 + S_update

    return mu, S



class GSM:
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

        
    def fit(self, key, mean=None, cov=None, batch_size=2, niter=5000, nprint=10, verbose=True, check_goodness=True, monitor=None, retries=10):
        """
        Main function to fit a multivariate Gaussian distribution to the target

        Inputs:
          key: Random number generator key (jax.random.PRNGKey)
          mean : Optional, initial value of the mean. Expected None or array of size D. Default=0.
          cov : Optional, initial value of the covariance matrix. Expected None or array of size DxD. Default=identity
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
                    mean_new, cov_new = gsm_update(samples, vs, mean, cov)
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
                nan_update.append(j)
            else:
                is_good = True
            return is_good
        except:
            return is_good
