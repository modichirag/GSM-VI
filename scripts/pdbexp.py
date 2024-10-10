import numpy as np
import sys, os
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
from jax import config
config.update("jax_enable_x64", True)

from imports import *

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--alg', type=str, help='which algorithm to run')
parser.add_argument('--modeln', type=int, help='model number for PosteriorDB')
#arguments for GSM
parser.add_argument('--seed', type=int, default=99, help='seed between 0-999, default=99')
parser.add_argument('--niter', type=int, default=1001, help='number of iterations in training')
parser.add_argument('--batch', type=int, default=2, help='batch size, default=2')
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
parser.add_argument('--reg', type=float, default=1e-2, help='regularizer for ngd and lsgsm')
parser.add_argument('--lambdat', type=int, default=0, help='which regularizer to use')
parser.add_argument('--modeinit', type=int, default=0, help='initialize from mode')
parser.add_argument('--lbfgsinit', type=int, default=0, help='initialize from lbfgs')
parser.add_argument('--scaleinit', type=float, default=1., help='scale the initial covariance')
#arguments for path name
parser.add_argument('--suffix', type=str, default="", help='suffix, default=""')


sys.path.append('/mnt/home/cmodi/Research/Projects/posterior_database/')
from posteriordb import BSDB
from jax_utils import jaxify_bs


print()
args = parser.parse_args()
lr, batch_size = args.lr, args.batch
alg = args.alg

def setup_pdb(model_n):
    
    model = BSDB(model_n)
    D = model.dims
    lpjax, lp_itemjax = jaxify_bs(model)   
    lp = model.lp
    lp_g = lambda x: model.lp_g(x)[1]
    lpjaxsum = jit(lambda x: jnp.sum(lpjax(x)))
    try:
        ref_samples = model.samples_unc.copy()
    except Exception as e:
        print(e)
        ref_samples = None

    return D, lpjaxsum, lp, lp_g, ref_samples


D, lpjax, lp, lp_g, ref_samples = setup_pdb(args.modeln)

basepath = "/mnt/ceph/users/cmodi/ls-gsm/"
path = f"{basepath}/PDB_{args.modeln}/"
os.makedirs(path, exist_ok=True)
print(f"Parent folder in : {path}")
print("For algorithm : ", alg)
print("For lr and batch : ", lr, batch_size)
if ref_samples is not None: np.save(f"/{path}/ref_samples", ref_samples)

#####
monitor = Monitor(batch_size=128, ref_samples=ref_samples, store_params_iter=20,
                  plot_samples=False, savepoint=1000)
seed = args.seed
key = random.PRNGKey(seed)
np.random.seed(seed)
x0 = np.random.random(D).astype(np.float64)*0.1
cov0 = np.identity(D) * args.scaleinit
suffix=''

if args.modeinit:
    suffix="-modeinit"
    x0, _, res = lbfgs_init(x0, lp, lp_g)
    monitor.offset_evals = res.nfev
    print(f'lbfgs output : \n{res}\n')

if args.lbfgsinit:
    suffix="-lbfgsinit"
    x0, cov0, res = lbfgs_init(x0, lp, lp_g)
    monitor.offset_evals = res.nfev
    print(f'lbfgs output : \n{res}\n')

if args.scaleinit != 1: suffix += f'-scaleinit{args.scaleinit:0.2f}'
############################################    
if alg == 'gsm':
    gsm = GSM(D=D, lp=lpjax, lp_g=lp_g)
    path = f"{path}/{alg}/B{batch_size}{suffix}/S{seed}/"
    monitor.savepath = path
    print(f"save in : {path}")
    mean_fit, cov_fit = gsm.fit(key, batch_size=batch_size, mean=x0, cov=cov0,
                                niter=args.niter, monitor=monitor)

elif alg == 'advi':
    advi = ADVI(D=D, lp=lpjax)
    opt = optax.adam(learning_rate=lr)
    path = f"{path}/{alg}/B{batch_size}-lr{lr:0.3f}{suffix}/S{seed}/"       
    monitor.savepath = path
    print(f"save in : {path}")
    mean_fit, cov_fit, losses = advi.fit(key, opt, batch_size=batch_size, mean=x0, cov=cov0,
                                         niter=args.niter, monitor=monitor)

elif alg == 'ngd':
    ngd = NGD(D=D, lp=lpjax, lp_g=lp_g)
    path = f"{path}/{alg}/B{batch_size}-lr{lr:0.3f}-reg{args.reg:0.2e}{suffix}/S{seed}/"    
    monitor.savepath = path
    print(f"save in : {path}")
    mean_fit, cov_fit = ngd.fit(key, lr=args.lr, batch_size=batch_size, mean=x0, cov=cov0,
                                reg=args.reg, niter=args.niter, monitor=monitor)
    

elif alg == 'lsgsm':
    lsgsm = LS_GSM(D=D, lp=lpjax, lp_g=lp_g)
    reg = args.reg
    if reg == 0: reg = batch_size * D
    regf = setup_regularizer(reg)[args.lambdat]
    path = f"{path}/{alg}/B{batch_size}-lambdat{args.lambdat}-reg{args.reg:0.2f}{suffix}/S{seed}/"
    monitor.savepath = path
    print(f"save in : {path}")
    mean_fit, cov_fit = lsgsm.fit(key, regf=regf, batch_size=batch_size, mean=x0, cov=cov0,
                                niter=args.niter, monitor=monitor)
   

os.makedirs(path, exist_ok=True)
np.save(f"{path}/mean", mean_fit)
np.save(f"{path}/cov", cov_fit)
np.save(f"{path}/means", monitor.means)
np.save(f"{path}/covs", monitor.covs)
np.save(f"{path}/iparams", monitor.iparams)
np.save(f"{path}/nevals", monitor.nevals)
np.save(f"{path}/rkl", monitor.rkl)
np.save(f"{path}/fkl", monitor.fkl)

# plot final figures
from gsmvi import plotting
plotting.plot_loss(monitor.nevals, monitor.rkl, path, fname='rkl', logit=True, ylabel='Reverse KL')
plotting.plot_loss(monitor.nevals, monitor.fkl, path, fname='fkl', logit=True, ylabel='Forward KL')

qsamples = np.random.multivariate_normal(mean=mean_fit, cov=cov_fit, size=1000)
plotting.corner(qsamples[:500],
                savepath=f"{path}/",
                savename=f"corner", maxdims=5) 

plotting.compare_hist(qsamples, ref_samples=ref_samples[:1000],
                      savepath=f"{path}/",
                      savename=f"hist") 

