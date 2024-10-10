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
from gsmvi.gsm_mf import GSM_MF
from gsmvi.monitors import KLMonitor
#####


#####
def setup_model(D=10):
   
    # setup a Gaussian target distribution
    mean = np.random.random(D)
    L = np.random.normal(size = D**2).reshape(D, D)
    cov = np.matmul(L, L.T) + np.eye(D)*1e-3
    sigmasq = np.random.uniform(0, 2, D)
    cov = np.diag(sigmasq)
    model = dist.MultivariateNormal(loc=mean, covariance_matrix=cov)
    lp = jit(lambda x: jnp.sum(model.log_prob(x)))
    lp_g = jit(grad(lp, argnums=0))

    return model, mean, cov, lp, lp_g



if __name__=="__main__":
    
    D = 5
    model, mean, cov, lp, lp_g = setup_model(D=D)
    ref_samples = model.sample(random.PRNGKey(99), (1000,))

    niter = 100
    batch_size = 2
    gsm = GSM_MF(D=D, lp=lp, lp_g=lp_g)
    key = random.PRNGKey(99)
    monitor = KLMonitor(batch_size=32, ref_samples=ref_samples, checkpoint=10, savepoint=1000,\
                        savepath='./tmp/', plot_samples=True)
    mean_fit, cov_fit = gsm.fit(key, niter=niter, batch_size=batch_size, monitor=monitor)

    print()
    print("True mean : ", mean)
    print("Fit mean  : ", mean_fit)
    print()
    print("Check mean fit")
    print(np.allclose(mean, mean_fit))

    print()
    print("Check cov fit")
    print(cov)
    print(np.diag(cov))
    print(cov_fit)
    print(np.allclose(np.diag(cov), cov_fit))
