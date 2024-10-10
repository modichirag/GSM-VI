## Example for fitting a target Multivariate Gaussian distribution with GSM updates
## GSM distribution is initialized with LBFGS fit for the mean and covariance
## and the progress is monitored with a Monitor class. 

import numpy as np
import os, time
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
# import sys
# sys.path.append('../src/')
# from gsm import GSM
# from initializers import lbfgs_init
# sys.path.append('../utils/')
# from monitors import KLMonitor
from gsmvi.gsm import GSM
from gsmvi.monitors import KLMonitor
from gsmvi.initializers import lbfgs_init

#####
def setup_model(D=10):
   
    # setup a Gaussian target distribution
    mean = np.random.random(D)
    L = np.random.normal(size = D**2).reshape(D, D)
    cov = np.matmul(L, L.T) + np.eye(D)*1e-3
    model = dist.MultivariateNormal(loc=mean, covariance_matrix=cov)
    lp = jit(lambda x: jnp.sum(model.log_prob(x)))
    lp_g = jit(grad(lp, argnums=0))
    ref_samples = model.sample(random.PRNGKey(999), (1000,))

    return mean, cov, lp, lp_g, ref_samples


def gsm_fit(D, lp, lp_g, ref_samples, niter=1000):

    gsm = GSM(D=D, lp=lp, lp_g=lp_g)
    rng = random.PRNGKey(99)

    # gsm without initialization
    monitor = KLMonitor(batch_size=32, ref_samples=ref_samples, checkpoint=10,\
                        savepath='./tmp/', plot_samples=True)
    mean_fit, cov_fit = gsm.fit(rng, niter=niter, monitor=monitor)
        
    # setup gsm with initilization from LBFGS fit
    mean_init = np.ones(D)
    mean_init, cov_init, res = lbfgs_init(mean_init, lp, lp_g)
    print(f'lbfgs output : \n{res}\n')
    
    monitor = KLMonitor(batch_size=32, ref_samples=ref_samples, checkpoint=10, \
                        savepath='./tmp2/', plot_samples=True, \
                        offset_evals=res.nfev) # offset with lbfgs evals for correct accounting
    mean_fit, cov_fit = gsm.fit(rng, mean=mean_init, cov=cov_init, niter=niter, monitor=monitor)

    return mean_fit, cov_fit



if __name__=="__main__":
    
    D = 10
    mean, cov, lp, lp_g, ref_samples = setup_model(D=D)

    niter = 500
    mean_fit, cov_fit = gsm_fit(D, lp, lp_g, niter=niter, ref_samples=ref_samples)
    
    print()
    print("True mean : ", mean)
    print("Fit mean  : ", mean_fit)
    print()
    print("Check mean fit")
    print(np.allclose(mean, mean_fit))

    print()
    print("Check cov fit")
    print(np.allclose(cov, cov_fit))
