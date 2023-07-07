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
import optax

import numpyro
import numpyro.distributions as dist

# Import GSM
import sys
sys.path.append('../src/')
from advi import ADVI

#####


#####
def setup_model(D=10):
   
    # setup a Gaussian target distribution
    np.random.seed(0)
    mu = np.random.random(D)
    L = np.random.normal(size = D**2).reshape(D, D)
    cov = np.matmul(L, L.T) + np.eye(D)*1e-3
    model = dist.MultivariateNormal(loc=mu, covariance_matrix=cov)
    lp = jit(lambda x: jnp.sum(model.log_prob(x)))
    lp_g = jit(grad(lp, argnums=0))

    return mu, cov, lp, lp_g


def advi_fit(D, lp, lp_g, niter=1000):

    advi = ADVI(D=D, lp=lp)
    key = random.PRNGKey(99)
    opt = optax.adam(learning_rate=1e-2)
    mu_fit, cov_fit, losses = advi.fit(key, opt, niter=niter)

    return mu_fit, cov_fit



if __name__=="__main__":
    
    D = 5
    mu, cov, lp, lp_g = setup_model(D=D)

    niter = 20000
    mu_fit, cov_fit = advi_fit(D, lp, lp_g, niter=niter)

    print()
    print("Check mean fit")
    print(np.allclose(mu, mu_fit))
    print(mu/mu_fit)

    print()
    print("Check cov fit")
    print(np.allclose(cov, cov_fit))
    print(cov/cov_fit)
