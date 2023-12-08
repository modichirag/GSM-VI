## A basic example for fitting a target Multivariate Gaussian distribution with GSM updates

## Uncomment the following lines if you run into memory issues with JAX
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

import numpy as np
import jax.numpy as jnp
from jax import jit, grad, random
import optax
import numpyro.distributions as dist

from gsmvi.advi import ADVI

def setup_model(D=10):
   
    # setup a Gaussian target distribution
    mean = np.random.random(D)
    L = np.random.normal(size = D**2).reshape(D, D)
    cov = np.matmul(L, L.T) + np.eye(D)*1e-3
    model = dist.MultivariateNormal(loc=mean, covariance_matrix=cov)
    return model 


if __name__=="__main__":
    
    ###
    # setup a toy Gaussia model and extract log-prob
    D = 4
    model =  setup_model(D=D)
    mean, cov = model.loc, model.covariance_matrix
    lp = jit(lambda x: jnp.sum(model.log_prob(x)))

    ###
    # Fit with advi
    niter = 10000
    lr = 1e-2
    batch_size = 16
    advi = ADVI(D=D, lp=lp)
    key = random.PRNGKey(99)
    opt = optax.adam(learning_rate=lr)
    mean_fit, cov_fit, losses = advi.fit(key, opt, batch_size=batch_size, niter=niter)

    print("\nTrue mean : ", mean)
    print("Fit mean  : ", mean_fit)

