import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, random, grad
from functools import partial
import os
from . import plotting

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
    #minv = jnp.linalg.pinv(m)
    res = ltpsinv@x
    #second_term = res.T@minv@res
    #mltpsinv = jnp.linalg.solve(m, ltpsinv)
    second_term = res.T@jnp.linalg.solve(m, res)
    
    logexp = -0.5 * (first_term - second_term)
    logdet = -0.5 * (jnp.linalg.slogdet(m)[1] + jnp.sum(jnp.log(psi))) #jnp.log(jnp.linalg.det(m)*jnp.prod(psi))
    logp = logexp + logdet - 0.5*D*jnp.log(2*jnp.pi)
    return logp


def monitor_lr(monitor, i, params, lp, key, nevals, force_save=False):

    mean, psi, llambda = params
    key, key_sample = random.split(key)
    np.random.seed(key_sample[0])
    monitor.nevals.append(monitor.offset_evals + nevals)

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

    if (i%monitor.store_params_iter ==0 ) :
        monitor.means.append(mean)
        monitor.llambdas.append(llambda)
        monitor.psis.append(psi)
        monitor.iparams.append(i)

    #save
    if ((i%monitor.savepoint == 0) or force_save) & (monitor.savepath is not None):

        print("Savepoint: saving current fit, loss and diagnostic plots")
        
        os.makedirs(monitor.savepath, exist_ok=True)
        np.save(f"{monitor.savepath}/mean_fit", mean)
        np.save(f"{monitor.savepath}/psi_fit", psi)
        np.save(f"{monitor.savepath}/llambda_fit", llambda)
        np.save(f"{monitor.savepath}/means", monitor.means)
        np.save(f"{monitor.savepath}/psis", monitor.psis)
        np.save(f"{monitor.savepath}/llambdas", monitor.llambdas)
        np.save(f"{monitor.savepath}/iparams", monitor.iparams)
        np.save(f"{monitor.savepath}/nevals", monitor.nevals)
        np.save(f"{monitor.savepath}/times", monitor.times)
        np.save(f"{monitor.savepath}/rkl", monitor.rkl)
        if monitor.ref_samples is not None:
            np.save(f"{monitor.savepath}/fkl", monitor.fkl)
            try:
                if monitor.plot_loss: plotting.plot_loss(monitor.nevals, monitor.fkl, monitor.savepath, fname='fkl', logit=True)
            except Exception as e: print(e)

        if monitor.plot_loss:
            try: plotting.plot_loss(monitor.nevals, monitor.rkl, monitor.savepath, fname='rkl', logit=True)
            except Exception as e: print(e)

        if monitor.plot_samples:
            try:
                batch_size = 1000
                eps = np.random.normal(0, 1, size=(batch_size, D))
                z = np.random.normal(0, 1, size=(batch_size, K))
                qsamples = mean + psi**0.5 * eps + (llambda@z.T).T
                # plotting.corner(qsamples[:500],
                #                 savepath=f"{monitor.savepath}/",
                #                 savename=f"corner{i}", maxdims=5)

                if force_save: savename = 'hist'
                else: savename = f'hist{i}'
                plotting.compare_hist(qsamples, ref_samples=monitor.ref_samples[:1000],
                                savepath=f"{monitor.savepath}/",
                                savename=savename)
            except Exception as e:
                print(f"Exception occured in plotting samples in monitor : {e}.\nSkip")

    monitor.offset_evals = monitor.nevals[-1]

