import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad, random
from numpyro.distributions import MultivariateNormal
import optax
      
class ADVI():
    """
    Class for fitting a multivariate Gaussian distribution with dense covariance matrix
    by maximizing ELBO.
    """
    
    def __init__(self, D, lp):
        """
        Inputs:
          D: (int) Dimensionality (number) of parameters.
          lp : Function to evaluate target log-probability distribution
               whose gradient can be evaluated with jax.grad(lp)
        """
        self.D = D
        self.lp = lp
        self.idx_tril = jnp.stack(jnp.tril_indices(D)).T
        
    def scales_to_cov(self, scales):
        scale_tril = jnp.zeros((self.D, self.D))
        scale_tril = scale_tril.at[self.idx_tril[:, 0], self.idx_tril[:, 1]].set(scales)
        cov = np.matmul(scale_tril, scale_tril.T)
        return cov

    def neg_elbo(self, params, key, batch_size):
        """
        Internal function to evaluate negative-ELBO which is the loss function 
        """
        loc, scales = params
        scale_tril = jnp.zeros((self.D, self.D))
        scale_tril = scale_tril.at[self.idx_tril[:, 0], self.idx_tril[:, 1]].set(scales)
        q = MultivariateNormal(loc=loc, scale_tril=scale_tril)
        #
        samples = q.sample(key, (batch_size,))
        logl = jnp.sum(self.lp(samples))
        logq = jnp.sum(q.log_prob(samples))
        elbo = logl - logq
        negelbo = -1. * elbo
        return negelbo

    def fit(self, key, opt, mean=None, cov=None, batch_size=8, niter=1000, nprint=10, monitor=None):
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
        
        lossf = jit(self.neg_elbo, static_argnums=(2))

        @jit
        def opt_step(params, opt_state, key):
            loss, grads = jax.value_and_grad(lossf, argnums=0)(params, key, batch_size=batch_size)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        if mean is None:
            mean = jnp.zeros(self.D)
        if cov is None:
            cov = np.identity(self.D)
            
        # Optimization is done on unconstrained Cholesky factors of covariance matrix
        L = np.linalg.cholesky(cov)
        scales = jnp.array(L[np.tril_indices(self.D)])            
        params = (mean, scales)

        # run optimization
        opt_state = opt.init(params)
        losses = []
        nevals = 1 

        for i in range(niter + 1):
            if(i%(niter//nprint)==0):
                print(f'Iteration {i} of {niter}')
            if monitor is not None:
                if (i%monitor.checkpoint) == 0:
                    mean = params[0]
                    cov = self.scales_to_cov( params[1]*1.)
                    monitor(i, [mean, cov], self.lp, key, nevals=nevals)
                    nevals = 0

            params, opt_state, loss = opt_step(params, opt_state, key)
            key, _ = random.split(key)
            losses.append(loss)
            nevals += batch_size


        # Convert back to mean and covariance matrix
        mean = params[0]
        cov = self.scales_to_cov( params[1]*1.)
        if monitor is not None:
            monitor(i, [mean, cov], self.lp, key, nevals=nevals)
            
        return mean, cov, losses
