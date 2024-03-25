# This test makes sure that jax implments reparameterization gradients.
import numpy as np
import jax.numpy as jnp
from jax import grad
import jax
from numpyro.distributions import MultivariateNormal

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

# enable 16 bit precision for jax                                                                                                                                                                                                                                                                                            
from jax import config
config.update("jax_enable_x64", True)


def vanilla_grad(loc, cov, key, target):
    dist = MultivariateNormal(loc, cov)
    sample = dist.sample(key)
    kl = grad(dist.log_prob)(sample) - grad(target.log_prob)(sample)
    kl = jnp.sum(kl**2)**0.5
    return kl

def reparam_grad(loc, cov, key, target):
    dist = MultivariateNormal(loc, cov)
    D = loc.size
    L = jnp.linalg.cholesky(cov)
    eps =  MultivariateNormal(jnp.zeros(D), jnp.eye(D)).sample(key)
    sample = loc + jnp.dot(L, eps)
    kl = grad(dist.log_prob)(sample) - grad(target.log_prob)(sample)
    kl = jnp.sum(kl**2)**0.5
    return kl

def stop_grad(loc, cov, key, target):
    dist = MultivariateNormal(loc, cov)
    sample = jax.lax.stop_gradient(dist.sample(key)) # stop gradient
    kl = grad(dist.log_prob)(sample) - grad(target.log_prob)(sample)
    kl = jnp.sum(kl**2)**0.5
    return kl



def test(D=2):
    
    loc = np.random.random(D)
    L = np.random.normal(size = D**2).reshape(D, D)
    cov = np.matmul(L, L.T) + np.eye(D)*1e-2
    target = MultivariateNormal(loc=loc, covariance_matrix=cov)

    g_vanilla = grad(vanilla_grad, argnums=(0, 1))
    g_reparam = grad(reparam_grad, argnums=(0, 1))
    g_stop = grad(stop_grad, argnums=(0, 1))
    

    key = jax.random.PRNGKey(0)
    loc = np.random.random(D)
    L = np.random.normal(size = D**2).reshape(D, D)
    cov = np.matmul(L, L.T) + np.eye(D)*1e-2

    g0 = g_vanilla(loc, cov, key, target)
    g1 = g_reparam(loc, cov, key, target)
    g2 = g_stop(loc, cov, key, target)

    print("vanilla gradient\n", g0)
    print("reparameterization gradient\n", g1)
    print("stop gradient\n", g2)

    print(np.allclose(g0[0], g1[0]))
    print(np.allclose(g0[1], g1[1]))
    
    # assert (g0[0] == g1[0]).all()
    # assert (g0[1].flatten() == g1[1].flatten()).all()
    # except Exception as e:
    #     print("exception : ", e)
    #     print(g0[1], g1[1])


test()
