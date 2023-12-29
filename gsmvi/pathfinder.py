import numpy as np
import jax
from numpyro.distributions import MultivariateNormal
from scipy_lbfgs_wrapper import minimize_lbfgsb

class Trajectory():
    """
    Callback for saveing the trajectory of LBFGS algorithm
    """
    def __init__(self):
        self.trajectory_x = []
        self.trajectory_hessinv = []
        self.trajectory_f = []
        self.length = 0 
        
    def __call__(self, res):
        """
        Append the position, Hessian_inverse and function value on LBFGS trajectory

        Inputs:
          res : OptimizeResult object from LBFGS 
                expected to have x, hess_inv and fun as attributes atleast
        """
        self.trajectory_x.append(res.x*1.)
        self.trajectory_hessinv.append(res.hess_inv.todense())
        self.trajectory_f.append(res.fun)
        self.length += 1

    def reset(self):
        self.trajectory_x = []
        self.trajectory_hessinv = []
        self.trajectory_f = []
        self.length = 0
        
    def to_array(self):
        self.trajectory_x = np.array(self.trajectory_x)
        self.trajectory_hessinv = np.array(self.trajectory_hessinv)
        self.trajectory_f = np.array(self.trajectory_f)
  


class Pathfinder():
    """
    Implement pathfinder algorithm wrapping over Scipy's implementation of LBFGS
    """

    def __init__(self, D, lp, lp_g):
        """
        Inputs:
          D: (int) Dimensionality (number) of parameters
          lp : Function to evaluate target log-probability distribution. 
          lp_g : Function to evaluate gradient of target log-probability distribution. 
        """
        self.D = D
        self.lp = lp
        self.lp_g = lp_g

    # if using jax, can jit this
    def _estimate_KL(self, mu, cov, batch_kl, key):
        """
        Estimate KL divergence between Gaussian distribution (mu, cov) and target distribution
        """
        key, key_sample = jax.random.split(key)
        #samples = np.random.multivariate_normal(mean=mu, cov=cov, size=batch_kl)
        q = MultivariateNormal(loc=mu, covariance_matrix=cov)    
        samples = q.sample(key_sample, (batch_kl,))
        logl = np.mean(self.lp(samples))
        logq = np.mean(q.log_prob(samples))
        kl = logq - logl
        return key, kl


    def fit(self, key, x0=None, num_samples=200, maxiter=1000, batch_kl=32, return_trajectory=False, verbose=True):
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

        # Negate lp as LBFGS will minimize the cost function
        f = lambda x: -self.lp(x)
        f_g = lambda x: -self.lp_g(x)

        if x0 is None:
            x0 = np.zeros(self.D)

        # Optimize with LBFGS
        t = Trajectory()
        res = minimize_lbfgsb(f, x0, jac=f_g, callback=t, maxcor=100, maxiter=maxiter)        
        if verbose : print(f"Output of LBFGS run is \n{res}")

        # Estimate KL at all points. Proposed optimization- to save memory,
        # this can be made part of callback, so we don't have to save entire trajecotry but only best point
        t.kls = []
        for i in range(t.length):
            key, kl = self._estimate_KL(t.trajectory_x[i], t.trajectory_hessinv[i], batch_kl, key)
            t.kls.append(kl)

        t.kls = np.array(t.kls)
        t.to_array()
        best_i = np.argmin(t.kls)
        if verbose: print (f"KL minimized at iteration {best_i}")
        mu = t.trajectory_x[best_i]
        cov = t.trajectory_hessinv[best_i]
        
        if return_trajectory:
            return mu, cov, t
        
        else:
            return mu, cov

