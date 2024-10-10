import numpy as np
import sys, os, time
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
MultivariateNormal = dist.MultivariateNormal
import optax

# Import GSM
from gsmvi.gsm import GSM
from gsmvi.bbvi import ADVI
from gsmvi.bam import BAM
from gsmvi.ngd import NGD
from gsmvi.initializers import lbfgs_init
from gsmvi.monitors import KLMonitor as Monitor
from gsmvi.monitors import Monitor_Flow
#from gsmvi.flowvi import FlowVI
#from gsmvi.realNVP import AffineCoupling, RealNVP
from setup_regs import setup_regularizer


def setup_gauss_model(D, seed=123, noise=1e-2, mean=None, rank=None):
    np.random.seed(seed)
    if mean is None:
        mean = np.random.random(D)
    if rank is None:
        rank = D
    L = np.random.normal(size = D*rank).reshape(D, rank)
    cov = np.matmul(L, L.T) + np.eye(D)*noise
    model = dist.MultivariateNormal(loc=mean, covariance_matrix=cov)
    lp = jit(lambda x: jnp.sum(model.log_prob(x)))
    lp_g = jit(grad(lp, argnums=0))
    ref_samples = np.random.multivariate_normal(mean, cov, 5000)
    return mean, cov, lp, lp_g, ref_samples



def setup_good_model(D=10, rank=4, seed=123):

    # setup a Gaussian target distribution                                                                                                                    
    np.random.seed(seed)
    mean = np.random.random(D)
    L = np.random.normal(size = D*rank).reshape(D, rank)/(D*rank)**0.5
    cov = np.matmul(L, L.T) + np.diag(np.random.random(D))
    model = dist.MultivariateNormal(loc=mean, covariance_matrix=cov)
    lp = jit(lambda x: jnp.sum(model.log_prob(x)))
    lp_g = jit(grad(lp, argnums=0))
    ref_samples = np.random.multivariate_normal(mean, cov, 10000)
    return model, mean, cov, lp, lp_g, ref_samples


def setup_model_nonorm(D=10, rank=4, seed=123):

    # setup a Gaussian target distribution                                                                                                                    
    np.random.seed(seed)
    mean = np.random.random(D)
    L = np.random.normal(size = D*rank).reshape(D, rank)
    cov = np.matmul(L, L.T) + np.diag(np.random.random(D))
    model = dist.MultivariateNormal(loc=mean, covariance_matrix=cov)
    lp = jit(lambda x: jnp.sum(model.log_prob(x)))
    lp_g = jit(grad(lp, argnums=0))
    ref_samples = np.random.multivariate_normal(mean, cov, 10000)
    return model, mean, cov, lp, lp_g, ref_samples

def setup_model_dnorm(D=10, rank=4, seed=123):

    # setup a Gaussian target distribution                                                                                                                    
    np.random.seed(seed)
    mean = np.random.random(D)
    L = np.random.normal(size = D*rank).reshape(D, rank)/(D)**0.5
    cov = np.matmul(L, L.T) + np.diag(np.random.random(D))
    model = dist.MultivariateNormal(loc=mean, covariance_matrix=cov)
    lp = jit(lambda x: jnp.sum(model.log_prob(x)))
    lp_g = jit(grad(lp, argnums=0))
    ref_samples = np.random.multivariate_normal(mean, cov, 10000)
    return model, mean, cov, lp, lp_g, ref_samples

def setup_model_ranknorm(D=10, rank=4, seed=123):

    # setup a Gaussian target distribution                                                                                                                    
    np.random.seed(seed)
    mean = np.random.random(D)
    L = np.random.normal(size = D*rank).reshape(D, rank)/(rank)**0.5
    cov = np.matmul(L, L.T) + np.diag(np.random.random(D))
    model = dist.MultivariateNormal(loc=mean, covariance_matrix=cov)
    lp = jit(lambda x: jnp.sum(model.log_prob(x)))
    lp_g = jit(grad(lp, argnums=0))
    ref_samples = np.random.multivariate_normal(mean, cov, 10000)
    return model, mean, cov, lp, lp_g, ref_samples



def setup_model(D=10, rank=4, seed=123):

    # setup a Gaussian target distribution                                                                                                                    
    np.random.seed(seed)
    mean = np.random.random(D)
    L = np.random.normal(size = D*rank).reshape(D, rank)
    cov = np.matmul(L, L.T) + np.diag(np.random.normal(1, 1, D)*1e-1+1)
    model = dist.MultivariateNormal(loc=mean, covariance_matrix=cov)
    lp = jit(lambda x: jnp.sum(model.log_prob(x)))
    lp_g = jit(grad(lp, argnums=0))
    ref_samples = np.random.multivariate_normal(mean, cov, 10000)
    return model, mean, cov, lp, lp_g, ref_samples

