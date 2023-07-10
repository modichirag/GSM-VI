import jax
import jax.numpy as jnp
from blackjax.optimizers.lbfgs import lbfgs_inverse_hessian_formula_1
from blackjax.kernels import pathfinder

class Pathfinder():
    """
    Wrapper class for Pathfinder algorithm implemented in blakcjax
    """

    def __init__(self, D, lp):
        """
        Inputs:
          D: (int) Dimensionality (number) of parameters
          lp : Function to evaluate target log-probability distribution. 
               whose gradient can be evaluated with jax.grad(lp)
        """
        self.D = D
        self.lp = lp


    def fit(self, key, x0=None, num_samples=200, max_iter=1000, return_path=False):
        """
        Main function to fit a multivariate Gaussian distribution to the target

        Inputs:
          key: Random number generator key (jax.random.PRNGKey)
          x0 : Optional, starting point for LBFGS optimization
          num_samples : Optional, Number of gradient samples to keep for approximating inverse Hessian
          max_iter : Optional, int. Max number of steps for LBFGS optimization
          return_path : Optional, bool. Keep fixed to False and current blackjax implementation results in error if true

        Returns:
          mu : Array of shape D, fit of the mean
          cov : Array of shape DxD, fit of the covariance matrix
        """
        
        if x0 is None:
            x0 = jnp.zeros(self.D)
        if return_path :
            try:
                finder = pathfinder(key, self.lp, num_samples=200, max_iter=1000, return_path=True)
                path = finder.init(x0)
                best_i = jnp.argmax(path.elbo)        
                print("Best ELBO at : ", best_i)
                state = jax.tree_map(lambda x: x[best_i], path)
                
                mu = state.position
                cov =  lbfgs_inverse_hessian_formula_1(state.alpha, state.beta, state.gamma)
                return mu, cov, state, path
                
            except Exception as e:
                print("Exception in running pathfinder with return_path flag :\n", e)
                
        finder = pathfinder(key, self.lp, num_samples=num_samples, max_iter=max_iter)
        state = finder.init(x0)
        
        mu = state.position
        cov =  lbfgs_inverse_hessian_formula_1(state.alpha, state.beta, state.gamma)
        
        return mu, cov, state


