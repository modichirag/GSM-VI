import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, random, grad
from functools import partial

@jit
def get_diag(U, V):
    """Return diagonal of U@V.T"""
    return jax.vmap(jnp.dot, in_axes=[0, 0])(U, V)


@jit
def det_cov_lr(psi, llambda):
    m = (llambda.T*(1/psi))@llambda
    m = np.identity(m.shape[0]) + m
    return jnp.linalg.det(m)*jnp.prod(psi)

@jit
def logp_lr(y, mean, psi, llambda):

    D, K = llambda.shape
    x = y - mean
    
    first_term = jnp.dot(x, x/psi)
    ltpsinv = llambda.T*(1/psi)
    m = jnp.identity(K) + ltpsinv@llambda
    minv = jnp.linalg.pinv(m)
    res = ltpsinv@x
    second_term = res.T@minv@res
    
    logexp = -0.5 * (first_term - second_term)
    logdet = -0.5 * jnp.log(jnp.linalg.det(m)*jnp.prod(psi))
    logp = logexp + logdet - 0.5*D*jnp.log(2*jnp.pi)
    return logp


def monitor_lr(monitor, i, params, lp, key, nevals):

    mean, psi, llambda = params
    key, key_sample = random.split(key)
    np.random.seed(key_sample[0])

    try:
        D, K = llambda.shape
        batch_size = monitor.batch_size
        eps = np.random.normal(0, 1, size=(batch_size, D))
        z = np.random.normal(0, 1, size=(batch_size, K))
        qsamples = mean + psi**0.5 * eps + (llambda@z.T).T

        func = partial(logp_lr, mean=mean, psi=psi, llambda=llambda)
        qprob = jax.vmap(func, in_axes=[0])(qsamples)
        pprob = lp(qsamples)
        monitor.rkl.append((qprob-pprob).mean())

        if monitor.ref_samples is not None:
            idx = np.random.permutation(monitor.ref_samples.shape[0])[:monitor.batch_size]
            psamples = monitor.ref_samples[idx]
            qprob = jax.vmap(func, in_axes=[0])(psamples)
            pprob = lp(psamples)
            monitor.fkl.append((pprob-qprob).mean())
    except Exception as e:
        print(f"Exception occured in monitor : {e}.\nAppending NaN")
        monitor.rkl.append(np.NaN)
        monitor.fkl.append(np.NaN)
        
    monitor.nevals.append(monitor.offset_evals + nevals)
    monitor.offset_evals = monitor.nevals[-1]

