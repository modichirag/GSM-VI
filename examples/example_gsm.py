## Most basic example for fitting a target Multivariate Gaussian distribution with GSM updates

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
from gsm import GSM
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



def gsm_fit(D, lp, lp_g, niter=1000):

    gsm = GSM(D=D, lp=lp, lp_g=lp_g)
    key = random.PRNGKey(99)
    mean_fit, cov_fit = gsm.fit(key, niter=niter)

    return mean_fit, cov_fit



if __name__=="__main__":
    
    D = 10
    mean, cov, lp, lp_g = setup_model(D=D)

    niter = 500
    mean_fit, cov_fit = gsm_fit(D, lp, lp_g, niter=niter)

    print("True mean : ", mean)
    print("Fit mean  : ", mean_fit)
    print()
    print("Check mean fit")
    print(np.allclose(mean, mean_fit))

    print()
    print("Check cov fit")
    print(np.allclose(cov, cov_fit))
