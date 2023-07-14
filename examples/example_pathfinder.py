## Most basic example for fitting a target Multivariate Gaussian distribution with Pathfinder

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

# enable 16 bit precision for jax
from jax import config
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from jax import jit, grad, random

import numpyro
import numpyro.distributions as dist

# Import GSM
import sys
sys.path.append('../src/')
from pathfinder import Pathfinder
#####


#####
def setup_model(D=10):
   
    # setup a Gaussian target distribution
    mean = np.random.random(D)
    L = np.random.normal(size = D**2).reshape(D, D)
    cov = np.matmul(L, L.T) + np.eye(D)*1e-3
    model = dist.MultivariateNormal(loc=mean, covariance_matrix=cov)
    lp = jit(lambda x: jnp.sum(model.log_prob(x)))
    lp_g = jit(grad(lp, argnums=0))

    return mean, cov, lp, lp_g



def pathfinder_fit(D, lp, lp_g, maxiter=1000, batch_kl=32):

    finder = Pathfinder(D=D, lp=lp, lp_g=lp_g)
    key = random.PRNGKey(99)
    mean_fit, cov_fit, trajectory = finder.fit(key, maxiter=maxiter, return_trajectory=True)
    
    return mean_fit, cov_fit



if __name__=="__main__":
    
    D = 5
    mean, cov, lp, lp_g = setup_model(D=D)

    maxiter = 1000 
    batch_kl = 64
    mean_fit, cov_fit = pathfinder_fit(D, lp, lp_g, maxiter=maxiter, batch_kl=batch_kl)

    print()
    print("True mean : ", mean)
    print("Fit mean  : ", mean_fit)
    print()
    print("True covariance matrix : \n", cov)
    print("Fit covariance matrix  : \n", cov_fit)
