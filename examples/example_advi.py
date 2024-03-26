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
from gsmvi.bbvi import ADVI
from gsmvi.monitors import KLMonitor
#####


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

    return model, mean, cov, lp, lp_g


if __name__=="__main__":
    
    D = 5
    np.random.seed(100)
    model, mean, cov, lp, lp_g = setup_model(D=D)
    ref_samples = model.sample(random.PRNGKey(99), (1000,))

    niter = 500
    lr = 5e-3
    batch_size = 16

    alg = ADVI(D=D, lp=lp, jit_compile=True)
    key = random.PRNGKey(99)
    opt = optax.adam(learning_rate=lr)
    monitor = KLMonitor(batch_size=32, ref_samples=ref_samples, checkpoint=10, savepoint=5000,\
                        savepath='./tmp/', plot_samples=True)
    mean_fit, cov_fit, losses = alg.fit(key, opt, batch_size=batch_size, niter=niter, monitor=monitor)


    print()
    print("True mean : ", mean)
    print("Fit mean  : ", mean_fit)
    print()
    print("True covariance matrix : \n", cov)
    print("Fit covariance matrix  : \n", cov_fit)
