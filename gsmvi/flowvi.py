import numpy as np
import jax.numpy as jnp
from jax import jit, grad, random
import equinox as eqx


class FlowVI():
    """
    Class for fitting a RealNVP to a target distribution by maximizing ELBO.
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
    

    @eqx.filter_jit
    def rkl_loss(self, model, key, batch_size):
        """Internal function to evaluate negative-ELBO, the loss function"""
        samples = model.sample(key, batch_size)
        logl = jnp.sum(self.lp(samples))
        logq = jnp.sum(model.log_prob(samples))
        elbo = logl - logq
        negelbo = -1. * elbo
        return negelbo


    def fit(self, key, model, opt, batch_size=8, niter=1000, nprint=10, retries=10):
        """
        Main function to fit a multivariate Gaussian distribution to the target

        Arguments:
            key: Random number generator key (jax.random.PRNGKey)
            key: Random number generator key (jax.random.PRNGKey)
            model: RealNVP model
            batch_size: Optional, int. Number of samples to match scores for at every iteration
            niter: Optional, int. Total number of iterations
            nprint: Optional, int. Number of iterations after which to print logs
            monitor: Optional. Function to monitor the progress and track different statistics for diagnostics.
                        Function call should take the input tuple (iteration number, [mean, cov], lp, key, number of grad evals).
                        Example of monitor class is provided in utils/monitors.py
        Returns:
          mu : Array of shape D, fit of the mean
          cov : Array of shape DxD, fit of the covariance matrix        

        """
        @eqx.filter_jit
        def opt_step(model, opt_state, key):
            loss, grads = eqx.filter_value_and_grad(self.rkl_loss)(model, key, batch_size)
            updates, opt_state = opt.update(grads, opt_state)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss

 
        # run optimization
        opt_state = opt.init(eqx.filter(model, eqx.is_array))
        losses = []
        nevals = 1

        for i in range(niter + 1):
            if(i%(niter//nprint)==0):
                print(f'Iteration {i} of {niter}')

            j = 0
            while True:         # Sometimes run crashes due to a bad sample. Avoid that by re-trying.
                try:
                    key, _ = random.split(key)
                    model, opt_state, loss = opt_step(model, opt_state, key)
                    nevals += batch_size
                    break
                except Exception as e:
                    if j < retries :
                        j += 1
                        print(f"Failed with exception {e}")
                        print(f"Trying again {j} of {retries}")
                    else : raise e
            losses.append(loss)

        return model, losses


