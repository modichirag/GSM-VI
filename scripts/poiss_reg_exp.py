import jax
import jax.random as jr

import numpy as np
import gpjax as gpx

import jax.scipy as jspy

# Enable Float64 for more stable matrix inversions.
from jax import config
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import install_import_hook
import matplotlib as mpl
import matplotlib.pyplot as plt

config.update("jax_enable_x64", True)

import numpyro
import numpyro.distributions as dist
MultivariateNormal = dist.MultivariateNormal
from numpyro.distributions import Normal
import optax

from gsmvi.gsm import GSM
from gsmvi.bbvi import ADVI, ADVI_LR, ADVI_Factorized

from gsmvi.bam_dc import BAM
from gsmvi.gsm import GSM
from gsmvi.pbam import PBAM # PBAM_fullcov
from gsmvi.pgsm import PGSM
from gsmvi.monitors import KLMonitor

from gsmvi.monitors import KLMonitor as Monitor
from time import time

import argparse
import os

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-D', type=int, help='dimension')
parser.add_argument('--rank', type=int, default=32, help='dimension')
parser.add_argument('--noise', type=float, default=0.01, help='rank')
parser.add_argument('--dataseed', type=int, default=0, help='seed between 0-999, default=0')
#arguments for GSM
parser.add_argument('--ranklr', type=int, default=0, help='rank of variational family')
parser.add_argument('--seed', type=int, default=99, help='seed between 0-999, default=99')
parser.add_argument('--niter', type=int, default=1001, help='number of iterations in training')
parser.add_argument('--batch', type=int, default=2, help='batch size, default=2')
parser.add_argument('--reg', type=float, default=1.0, help='regularizer for ngd and lsgsm')
parser.add_argument('--eta', type=float, default=1.2, help='eta factor for em updates')
parser.add_argument('--niter_em', type=int, default=501, help='which regularizer to use')
parser.add_argument('--nprint', type=int, default=100, help='number of times to print')
parser.add_argument('--tolerance', type=float, default=1e-5, help='regularizer for ngd and lsgsm')
parser.add_argument('--updatemode', type=str, default="current", help='initialziation of em updates')
parser.add_argument('--updateform', type=str, default="lawrence", help='lawrence or diana updates')
parser.add_argument('--postmean', type=int, default=0, help='number of times to print')
#args for monitor
parser.add_argument('--checkpoint', type=int, default=10, help='number of times to print')
parser.add_argument('--store_params_iter', type=int, default=50, help='number of times to print')
parser.add_argument('--savepoint', type=int, default=100, help='number of times to print')

parser.add_argument('--lr', type=float, default=1e-2, help='regularizer for ngd and lsgsm')

parser.add_argument('--schedule', type=float, default=0.5, help='scheduling learning rate of BAM')

parser.add_argument('--algorithm', type=str, default=0.5, help='bam, pbam, or advi')


#arguments for path name
parser.add_argument('--suffix', type=str, default="", help='suffix, default=""')
parser.add_argument('--cond', type=int, default=0, help='suffix, default=""')

print()
args = parser.parse_args()

if args.suffix != '': suffix = f"-{args.suffix}"
else: suffix = ""

D = args.D
rank = args.rank
niter = args.niter
lr = args.lr
batch_size = args.batch
nprint = args.nprint
seed = args.seed
algorithm = args.algorithm

#if rank > D:
#    print("Rank greater than dimensions, set to D/2")
#    rank = D//2


# Paths
print(D)
#basepath = '/mnt/ceph/users/dcai1/pbam/'
basepath = "./output/"


# LR + D
if args.cond == 0: path = f"{basepath}/Poisson-D{D}/{algorithm}/R{rank}-seed{args.dataseed}/"
# Full rank
elif args.cond == 1: path = f"{basepath}/Poisson-D{D}/{algorithm}/fullrank-seed{args.dataseed}/"
# Diagonal
elif args.cond == 2: path = f"{basepath}/Poisson-D{D}/{algorithm}/diag-seed{args.dataseed}/"

savepath = f'{path}/B{args.batch}-lr{args.lr:0.3f}{suffix}/'


os.makedirs(savepath, exist_ok=True)
print(f"Save results in {savepath}")

#########################################################################################################################

# Form log prob

# length scale
ls = 1
# signal variance
s_f = 1

kernel_rbf = lambda x, y: s_f**2 * jnp.exp(-0.5 * jnp.linalg.norm(x - y)**2 / ls**2)

kernel_rbf = jax.jit(kernel_rbf)

print(kernel_rbf(jnp.zeros(2), jnp.ones(2)))

#D = 1
D = 100
#xlim = jnp.linspace(0, 100, D)
xlim = jnp.linspace(-3, 3, D)

kernel_matrix = jax.vmap(jax.vmap(kernel_rbf, in_axes=(None, 0)), in_axes=(0, None))(xlim, xlim)+ 1e-6*np.eye(D)

print(jnp.linalg.cond(kernel_matrix))

key = jr.key(1234)

ref_samples = jr.multivariate_normal(key, mean=jnp.zeros(D), cov=kernel_matrix, shape=(50,))
#key, subkey = jr.split(key)

# log f ~ GP
g = jr.multivariate_normal(key, mean=jnp.zeros(D), cov=kernel_matrix) # + 1e-6*np.eye(D))
# Poisson rate
f = jnp.exp(g)
# observed counts
y = jr.poisson(key, lam=f)

print(f)
print(y)

plt.plot(xlim, f)
plt.scatter(xlim, y)
plt.savefig(f'{savepath}/data.png')


def lp(z):
    # global variables: data y, prior kernel_matrix
    lp = jspy.stats.multivariate_normal.logpdf(z, mean=jnp.zeros(D), cov=kernel_matrix)
    lp += jspy.stats.poisson.logpmf(y, jnp.exp(z)).sum()
    return lp

lp_g = jax.jit(jax.grad(lp, argnums=0))

lp_vmap = lambda x: jax.vmap(lp, in_axes=0)(x)
lp_g_vmap = lambda x: jax.vmap(lp_g, in_axes=0)(x)

#########################################################################################################################

key = jax.random.PRNGKey(2)
np.random.seed(args.seed)

mean = jnp.zeros(D)
psi = np.random.random(D)
llambda = np.random.normal(0, 1, size=(D, rank))

if algorithm == 'advi':

    print("Learning rate:", lr)

    # Use adam
    opt = optax.adam(learning_rate=lr)
    #opt = optax.adam(learning_rate=schedule)


    if args.cond == 0:
        # Run LR+D ADVI
        alg = ADVI_LR(D, rank, lp_vmap, jit_compile=True)

        monitor = KLMonitor(batch_size=32, ref_samples=ref_samples, checkpoint=10, store_params_iter=10, plot_samples=True, savepath=f'{savepath}/')

        meanfit_advi_lr, psi_advi_lr, lambda_advi_lr, losses_lr = alg.fit(key, opt, mean=mean, psi=psi, llambda=llambda,
                                                                          batch_size=batch_size, niter=niter, nprint=nprint, \
                                        monitor=monitor)

        covfit_advi_lr = lambda_advi_lr @ lambda_advi_lr.T + psi_advi_lr
        np.save(f'{savepath}/means', monitor.means)
        np.save(f'{savepath}/llambdas', monitor.llambdas)
        np.save(f'{savepath}/psis', monitor.psis)

    elif args.cond == 1:

        # Run full ADVI
        alg = ADVI(D, lp_vmap)
        monitor = KLMonitor(batch_size=32, ref_samples=ref_samples, checkpoint=10, store_params_iter=10, plot_samples=True, savepath=f'{savepath}/')

        meanfit_advi, covfit_advi, losses = alg.fit(key, opt, batch_size=batch_size, niter=niter, nprint=nprint, monitor=monitor)

        np.save(f'{savepath}/means', monitor.means)
        np.save(f'{savepath}/covs', monitor.covs)
    elif args.cond == 2:

        # Run factorized ADVI
        alg = ADVI_Factorized(D, lp_vmap)
        monitor = KLMonitor(batch_size=32, ref_samples=ref_samples, checkpoint=10, store_params_iter=10, plot_samples=True, savepath=f'{savepath}/')

        meanfit_advi_diag, covfit_advi_diag, losses_diag = alg.fit(key, opt, batch_size=batch_size, niter=niter, nprint=nprint, monitor=monitor)

        np.save(f'{savepath}/means', monitor.means)
        np.save(f'{savepath}/covs', monitor.covs)
elif algorithm == 'bam':

    # LR + D
    if args.cond == 0:

        pbam = PBAM(D, lp_vmap, lp_g_vmap)
        regf = lambda x: D * batch_size #1000#/(1+x)

        monitor = KLMonitor(batch_size=32, ref_samples=ref_samples, checkpoint=10, store_params_iter=10,  plot_samples=True, savepath=f'{savepath}/')

        meanfit_pbam2, psi2, llambda2 = pbam.fit(key, rank=rank, batch_size=batch_size, niter=niter, \
                                          regf=regf, nprint=nprint, \
                                          tolerance=1e-4, eta=1.0, niter_em=501, \
                                          print_convergence=False, monitor=monitor)

        covfit_pbam2 = np.diag(psi2) + llambda2@llambda2.T


    elif args.cond == 1:

        alg = BAM(D, lp_vmap, lp_g_vmap, use_lowrank=True)

        regf = lambda x: D * batch_size #1000#/(1+x)

        monitor = KLMonitor(batch_size=32, ref_samples=ref_samples,
                            checkpoint=10, store_params_iter=10,
                            plot_samples=False, savepath=f'{savepath}/')

        meanfit_bam, covfit_bam = alg.fit(key, batch_size=batch_size, niter=niter, regf=regf, nprint=nprint, \
                                        monitor=monitor, check_goodness=False)

    elif args.cond == 2:
        print("Diagonal BaM not implemented")

plt.figure(figsize=(7, 3))
plt.subplot(121)
plt.plot(monitor.nevals, np.abs(monitor.rkl))
plt.loglog()
plt.ylabel('reverse kl')
plt.subplot(122)
plt.plot(monitor.nevals, np.abs(monitor.fkl))
plt.loglog()
plt.ylabel('forward kl')
plt.savefig(f'{savepath}/loss.png')
plt.close()






































