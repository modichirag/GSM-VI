import jax
import jax.numpy as jnp
from blackjax.optimizers.lbfgs import lbfgs_inverse_hessian_formula_1
from blackjax.kernels import pathfinder

class Pathfinder():

    def __init__(self, D, lp):
        self.D = D
        self.lp = lp


    def fit(self, key, x0=None, num_samples=200, max_iter=1000, return_path=False):
        
        if x0 is None:
            x0 = jnp.zeros(self.D)
        print(x0)
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


