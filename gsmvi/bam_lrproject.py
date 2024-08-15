import jax
import jax.numpy as jnp
from jax import jit, random
from jax.scipy.linalg import sqrtm as sqrtm_jsp
from scipy.linalg import sqrtm as sqrtm_sp
#from numpyro.distributions import MultivariateNormal  ##Needed if sampling from numpyro dist below
import numpy as np
import scipy.sparse as spys
from jax.lib import xla_bridge
from em_lr_projection import fit_lr_gaussian

from ls_gsm import ls_gsm_update, ls_gsm_lowrank_update



class BAM_lrproject:
    """
    Wrapper class for using GSM updates to fit a distribution
    """
    def __init__(self, D, lp, lp_g, use_lowrank=False, jit_compile=True):
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
        self.use_lowrank = use_lowrank
        if use_lowrank:
            print("Using lowrank update")
        self.jit_compile = jit_compile
        if not jit_compile:
            print("Not using jit compilation. This may take longer than it needs to.")


    def fit(self, key, regf, rank, mean=None, cov=None,
            batch_size=2, niter=5000, nprint=10, n_project=128,
            verbose=True, check_goodness=True, monitor=None, retries=10, jitter=1e-6):
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

        llambda, psi = None, None
        nevals = 1

        if self.use_lowrank:
            update_function = ls_gsm_lowrank_update
        else:
            update_function = ls_gsm_update
        if self.jit_compile:
            update_function = jit(update_function)
                
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
                    reg = regf(i)
                    mean_new, cov_new = update_function(samples, vs, mean, cov, reg)
                    cov_new += np.eye(self.D) * jitter # jitter covariance matrix
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
                
            x = np.random.multivariate_normal(mean, cov, n_project)
            if psi is None: 
                psi = jnp.diag(jnp.diag(cov))
            if llambda is None: 
                llambda = np.linalg.eigh(cov)[1][:, :rank]
            _, llambda, psi = fit_lr_gaussian(x, rank, verbose=False,
                                                 mu=mean, llambda=llambda, psi=psi)
            cov = llambda@llambda.T + psi
        

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
