import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="pl'atform"
# enable 16 bit precision for jax
from jax import config
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from jax import jit, grad, random

import numpyro
import numpyro.distributions as dist
MultivariateNormal = dist.MultivariateNormal
import optax

# Import GSM
from gsmvi.gsm import GSM
from gsmvi.bbvi import ADVI
from gsmvi.ls_gsm import LS_GSM
from gsmvi.ngd import NGD
from gsmvi.initializers import lbfgs_init
from gsmvi.monitors import KLMonitor as Monitor
from setup_regs import setup_regularizer


def setup_gauss_model(D, seed=123, noise=1e-2, mean=None):
    np.random.seed(seed)
    if mean is None:
        mean = np.random.random(D)
    L = np.random.normal(size = D**2).reshape(D, D)
    cov = np.matmul(L, L.T) + np.eye(D)*noise
    model = dist.MultivariateNormal(loc=mean, covariance_matrix=cov)
    lp = jit(lambda x: jnp.sum(model.log_prob(x)))
    lp_g = jit(grad(lp, argnums=0))
    ref_samples = np.random.multivariate_normal(mean, cov, 5000)
    return mean, cov, lp, lp_g, ref_samples

