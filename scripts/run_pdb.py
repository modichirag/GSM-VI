import numpy as np
import sys, os
import matplotlib.pyplot as plt
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
import sys
sys.path.append('../../GSM-VI/src/')
from gsm import GSM
from advi import ADVI
from pathfinder import Pathfinder
from pathfinder_blackjax import Pathfinder as Pathfinder_blackjax
from initializers import lbfgs_init

sys.path.append('../../GSM-VI/utils/')
from monitors import KLMonitor
import plotting

sys.path.append('/mnt/home/cmodi/Research/Projects/posterior_database/')
from posteriordb import BSDB
from jax_utils import jaxify_bs


def fit(model_n, savepath, niter=5000, batch_size=8, lr=1e-3, seed=42):
    
    model = BSDB(model_n)
    D = model.dims
    lpjax, lp_itemjax = jaxify_bs(model)   
    lp = model.lp
    lp_g = lambda x: model.lp_g(x)[1]
    lpjaxsum = jit(lambda x: jnp.mean(lpjax(x)))
    ref_samples = model.samples_unc.copy()    

    np.random.seed(seed)
    rng = jax.random.PRNGKey(seed)
    x0 = model.samples_unc.copy()[0]
    #x0 = np.random.normal(0, 1, size=D)
    #x0, _, res = lbfgs_init(x0, lp, lp_g, maxiter=5)
    #print(res.nfev)
    mu_init, cov_init, res = lbfgs_init(x0, lp, lp_g)
    print("\nLBFGS result : ", res)
    
    results = {}
    monitor = KLMonitor(batch_size=100,
                        ref_samples=ref_samples,
                        savepath='./tmp/',
                        plot_samples=True,
                        savepoint=niter,
                        checkpoint=50) 

    # GSM without initialization
    alg = "gsm"
    print(f"\nFor algorithm {alg}")
    gsm = GSM(D=D, lp=lp, lp_g=lp_g)
    monitor.reset(savepath=f'{savepath}/{alg}/', offset_evals=0)
    mu, cov = gsm.fit(rng, mean=x0, niter=niter, batch_size=batch_size, monitor=monitor)
    results[alg] = {"mean":mu, "cov":cov, "loss":np.stack([monitor.nevals, monitor.bkl, monitor.fkl])}
    for key in results[alg]:
        np.save(f"{savepath}/{alg}_{key}", results[alg][key])
    
    # GSM with initialization
    alg = "gsm-lbinit"
    print(f"\nFor algorithm {alg}")
    monitor.reset(savepath=f'{savepath}/{alg}/', offset_evals=res.nfev) #lbfgs offset
    mu, cov = gsm.fit(rng, mean=mu_init, cov=cov_init, niter=niter, monitor=monitor)
    results[alg] = {"mean":mu, "cov":cov, "loss":np.stack([monitor.nevals, monitor.bkl, monitor.fkl])}
    for key in results[alg]:
        np.save(f"{savepath}/{alg}_{key}", results[alg][key])

    # ADVI
    alg = "advi"
    print(f"\nFor algorithm {alg}")
    opt = optax.adam(learning_rate=lr)
    advi = ADVI(D=D, lp=lpjaxsum)
    monitor.reset(savepath=f'{savepath}/{alg}/', offset_evals=0)
    mu, cov, losses = advi.fit(rng, mean=x0, opt=opt, niter=niter, monitor=monitor)
    results[alg] = {"mean":mu, "cov":cov, "loss":np.stack([monitor.nevals, monitor.bkl, monitor.fkl])}
    for key in results[alg]:
        np.save(f"{savepath}/{alg}_{key}", results[alg][key])
    np.save(f"{savepath}/{alg}_losses", losses)

    # ADVI with initialization
    alg = "advi-lbinit"
    print(f"\nFor algorithm {alg}")
    opt = optax.adam(learning_rate=lr)
    advi = ADVI(D=D, lp=lpjaxsum)
    monitor.reset(savepath=f'{savepath}/{alg}/', offset_evals=res.nfev) #lbfgs offset
    mu, cov, losses = advi.fit(rng, opt=opt, mean=mu_init, cov=cov_init, niter=niter, monitor=monitor)
    results[alg] = {"mean":mu, "cov":cov, "loss":np.stack([monitor.nevals, monitor.bkl, monitor.fkl])}
    for key in results[alg]:
        np.save(f"{savepath}/{alg}_{key}", results[alg][key])
    np.save(f"{savepath}/{alg}_losses", losses)

    # Pathfinder
    alg = "path_bjax"
    print(f"\nFor algorithm {alg}")
    finder = Pathfinder_blackjax(D, lp_itemjax)
    mu, cov, state, path = finder.fit(rng, x0=x0, maxiter=niter, return_path=True)
    results[alg] = {"mean":mu, "cov":cov, "loss":np.stack([1+np.arange(path.elbo.size), -path.elbo])}
    for key in results[alg]:
        np.save(f"{savepath}/{alg}_{key}", results[alg][key])
    best_elbo = state.elbo
    
    # Pathfinder
    alg = "path"
    print(f"\nFor algorithm {alg}")
    finder = Pathfinder(D, lp, lp_g)
    mu, cov, path = finder.fit(rng, x0=x0, maxiter=niter, return_trajectory=True)
    results[alg] = {"mean":mu, "cov":cov, "loss":np.stack([1+np.arange(path.kls.size), path.kls])}
    for key in results[alg]:
        np.save(f"{savepath}/{alg}_{key}", results[alg][key])

    # plot and compare
    samples = []
    lbls = []
    for alg in results:
        mean, cov = results[alg]['mean'], results[alg]['cov']
        ss = np.random.multivariate_normal(mean=mean, cov=cov, size=1000)
        samples.append(ss)
        lbls.append(alg)

    plotting.compare_hist(samples, ref_samples, lbls=lbls, savepath=savepath)
    plt.figure()
    for alg in results:
        plt.plot(results[alg]['loss'][0], results[alg]['loss'][1], label=alg)
    plt.axhline(-best_elbo, color='k', lw=1, ls="--")
    plt.grid(which='both')
    plt.loglog()
    plt.legend()
    plt.savefig(f"{savepath}/compare_loss.png")
    
if __name__=="__main__":
    
    model_n = int(sys.argv[1])
    savepath = f'/mnt/ceph/users/cmodi/polyakVI/pathfinder/PDB_{model_n}/'
    os.makedirs(savepath, exist_ok=True)
    fit(model_n, savepath)
    
