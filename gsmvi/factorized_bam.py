import jax
import jax.numpy as jnp
from jax import jit, random
import numpy as np
import numpyro.distributions as dist


def fg_bam_update(lp, lp_g, samples, mu0, S0, reg):
    """
    Returns updated mean and diagonal covariance matrix with factorized BaM updates.
    """
    B = samples.shape[0]

    zbar = jnp.mean(samples, axis=0)
    C = jnp.var(samples, axis=0)

    g = lp_g(samples)

    gbar = jnp.mean(g, axis=0)
    Gamma = jnp.var(g, axis=0)

    a = Gamma + jnp.square(gbar) / (1 + reg)
    b = 1 / reg
    c = -  (C + S0 / reg + jnp.square(mu0 - zbar) / (1 + reg))

    # print("gbar:", gbar)
    # print("Gamma:", Gamma)
    # print("a:", a)
    # print("b: ", b)
    # print("c:", c)

    S_1 = (-b + jnp.sqrt(jnp.square(b) - 4 * jnp.multiply(a, c))) / (2 * a)
    S_2 = (-b - jnp.sqrt(jnp.square(b) - 4 * jnp.multiply(a, c))) / (2 * a)
    S = jnp.maximum(S_1, S_2)

    mu = (zbar + jnp.multiply(S, gbar)) * reg + mu0
    mu = mu / (1 + reg)

    return mu, S
    

class FG_BAM:
    """
    Wrapper class for using factorized BaM updates to fit a distribution.
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
    
    def fit(self, key, regf, mean=None, cov_diag=None, batch_size=2, niter=5000, nprint=10, jitter=1e-6):
        """
        Main function to fit a factorized Gaussian distribution to the target.
        """
        if mean is None:
            mean = jnp.zeros(self.D)
        if cov_diag is None:
            cov_diag = jnp.ones(self.D)
        
        # params = (mean, cov_diag)
        # nevals = 1
        mean_all = np.empty((niter, self.D))
        cov_all = np.empty((niter, self.D))

        if nprint > niter: nprint = niter
        for i in range(niter):
            if (i%(niter//nprint) == 0): print(f'Iteration {i} of {niter}')
            mean_all[i, :] = mean
            cov_all[i, :] = cov_diag

            key, _ = random.split(key)        
            eps = dist.Normal(0, 1).expand([self.D]).sample(key, (batch_size, ))
            samples = mean + eps * jnp.sqrt(cov_diag)

            # if i == 0 : reg = regf(i)
            reg = regf(i)
            reg = max(reg, 1e-5)

            mean_new, cov_new = fg_bam_update(self.lp, self.lp_g, samples, mean, cov_diag, reg)

            cov_new += np.ones(self.D) * jitter
            # cov_new = (cov_new + cov_new) / 2  # symmetrize

            mean, cov_diag = mean_new, cov_new
        
        return mean, cov_diag, mean_all, cov_all



