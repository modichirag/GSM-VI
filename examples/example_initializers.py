## Example for fitting a target Multivariate Gaussian distribution with GSM and ADVI
## Variational distribution is initialized with LBFGS fit for the mean and covariance
## The progress is monitored with a Monitor class. 

## Uncomment the following lines if you run into memory issues with JAX
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import jit, grad, random
import numpyro.distributions as dist
import optax

# enable 16 bit precision for jax required for lbfgs initializer
from jax import config
config.update("jax_enable_x64", True)

from gsmvi.gsm import GSM
from gsmvi.advi import ADVI
from gsmvi.initializers import lbfgs_init
from gsmvi.monitors import KLMonitor

#####
def setup_model(D=10):
   
    # setup a Gaussian target distribution
    mean = np.random.random(D)
    L = np.random.normal(size = D**2).reshape(D, D)
    cov = np.matmul(L, L.T) + np.eye(D)*1e-3
    model = dist.MultivariateNormal(loc=mean, covariance_matrix=cov)
    return model 


#
def gsm_fit(D, lp, lp_g, mean_init, cov_init, lbfgs_res):
    print("Now fit with GSM")
    niter = 500
    batch_size = 1
    key = random.PRNGKey(99)
    monitor = KLMonitor(batch_size_kl=32, checkpoint=10, \
                        offset_evals=lbfgs_res.nfev) #note the offset number of evals

    gsm = GSM(D=D, lp=lp, lp_g=lp_g)
    mean_fit, cov_fit = gsm.fit(key, mean=mean_init, cov=cov_init, niter=niter, batch_size=batch_size, monitor=monitor)
    return mean_fit, cov_fit, monitor


#
def advi_fit(D, lp, lp_g, mean_init, cov_init, lbfgs_res):
    print("\nNow fit with ADVI")
    niter = 500
    lr = 1e-2
    batch_size = 1
    key = random.PRNGKey(99)
    opt = optax.adam(learning_rate=lr)
    monitor = KLMonitor(batch_size_kl=32, checkpoint=10, \
                        offset_evals=lbfgs_res.nfev) #note the offset number of evals

    advi = ADVI(D=D, lp=lp)
    mean_fit, cov_fit, losses = advi.fit(key, mean=mean_init, cov=cov_init, opt=opt, batch_size=batch_size, niter=niter, monitor=monitor)
    return mean_fit, cov_fit, monitor



if __name__=="__main__":
    
    ###
    # setup a toy Gaussia model and extracet score needed for GSM
    D = 16
    model =  setup_model(D=D)
    mean, cov = model.loc, model.covariance_matrix
    lp = jit(lambda x: jnp.sum(model.log_prob(x)))
    lp_g = jit(grad(lp, argnums=0))

    ###
    print("Initialize with LBFGS")
    mean_init = np.ones(D)     # setup gsm with initilization from LBFGS fit
    mean_init, cov_init, lbfgs_res = lbfgs_init(mean_init, lp, lp_g)
    print(f'LBFGS fit: \n{lbfgs_res}\n')
    
    mean_gsm, cov_gsm, monitor_gsm = gsm_fit(D, lp, lp_g, mean_init, cov_init, lbfgs_res)
    mean_advi, cov_advi, monitor_advi = advi_fit(D, lp, lp_g, mean_init, cov_init, lbfgs_res)

    # Check that the output is correct
    print("\nTrue mean : ", mean)
    print("Fit gsm  : ", mean_gsm)
    print("Fit advi  : ", mean_advi)


    # Check that the KL divergence decreases
    plt.plot(monitor_gsm.nevals, monitor_gsm.rkl, label='GSM')
    plt.plot(monitor_advi.nevals, monitor_advi.rkl, label='ADVI')
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Reverse KL")
    plt.savefig("monitor_kl.png")
