from blackjax.optimizers.lbfgs import lbfgs_inverse_hessian_formula_1
from blackjax.kernels import pathfinder

class Pathfinder():

    def __init__(self, D, lp):
        self.D = D
        self.lp = lp


    def fit(self, key, x0=None, num_samples=200, max_iter=1000):
        
        finder = pathfinder(key, lp, num_samples=num_samples, max_iter=max_iter)
            
        if x0 is None:
            x0 = jnp.zeros(self.D)

        state = finder.init(x0)
        
        mu = state.position
        cov =  lbfgs_inverse_hessian_formula_1(state.alpha, state.beta, state.gamma)
        
        return mu, cov, state
