import sys, os
import os

from gsmvi.bbvi import ADVI_LR
from gsmvi.monitors import KLMonitor

import matplotlib.pyplot as plt
from imports import *
jax.config.update('jax_platform_name', 'cpu')
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-D', type=int, help='dimension')
parser.add_argument('-rank', type=int, default=16, help='dimension')
parser.add_argument('-noise', type=float, default=0.01, help='rank')
parser.add_argument('--dataseed', type=int, default=0, help='seed between 0-999, default=0')
#arguments for GSM
parser.add_argument('--seed', type=int, default=99, help='seed between 0-999, default=99')
parser.add_argument('--niter', type=int, default=1001, help='number of iterations in training')
parser.add_argument('--batch', type=int, default=2, help='batch size, default=2')
parser.add_argument('--reg', type=float, default=1.0, help='regularizer for ngd and lsgsm')
parser.add_argument('--eta', type=float, default=1.2, help='regularizer for ngd and lsgsm')
parser.add_argument('--niter_em', type=int, default=101, help='which regularizer to use')
parser.add_argument('--nprint', type=int, default=100, help='number of times to print')
parser.add_argument('--tolerance', type=float, default=1e-4, help='regularizer for ngd and lsgsm')
parser.add_argument('--lr', type=float, default=1e-2, help='regularizer for ngd and lsgsm')
#args for monitor
parser.add_argument('--checkpoint', type=int, default=10, help='number of times to print')
parser.add_argument('--store_params_iter', type=int, default=50, help='number of times to print')
parser.add_argument('--savepoint', type=int, default=100, help='number of times to print')

#arguments for path name
parser.add_argument('--suffix', type=str, default="", help='suffix, default=""')
parser.add_argument('--badcond', type=int, default=0, help='suffix, default=""')

print()
args = parser.parse_args()
if args.suffix != '': suffix = f"-{args.suffix}"
else: suffix = ""
D = args.D
rank = args.rank
if rank > D:
    print("Rank greater than dimensions, set to D/2")
    rank = D//2

# Paths
print(D)
basepath = '/mnt/ceph/users/cmodi/pbam/'
if args.badcond: path = f"{basepath}/Gauss-D{D}-badcond/R{rank}-seed{args.dataseed}/"
else: path = f"{basepath}/Gauss-D{D}/R{rank}-seed{args.dataseed}/"
os.makedirs(path, exist_ok=True)
savepath = f'{path}/padvi/B{args.batch}-lr{args.lr:0.3f}{suffix}/'
os.makedirs(savepath, exist_ok=True)
print(f"Save results in {savepath}")
      
def setup_model(D=10, rank=4):

    # setup a Gaussian target distribution
    np.random.seed(args.dataseed)
    mean = np.random.random(D)
    L = np.random.normal(size = D*rank).reshape(D, rank)
    cov = np.matmul(L, L.T) + np.diag(np.random.normal(1, 1, D)*1e-1+1)
    model = dist.MultivariateNormal(loc=mean, covariance_matrix=cov)
    lp = jit(lambda x: jnp.sum(model.log_prob(x)))
    lp_g = jit(grad(lp, argnums=0))
    ref_samples = np.random.multivariate_normal(mean, cov, 10000)
    return model, mean, cov, lp, lp_g, ref_samples


if args.badcond: mean, cov, lp, lp_g, ref_samples = setup_gauss_model(D, rank) # Bad conditioning
else: model, mean, cov, lp, lp_g, ref_samples = setup_model(D, rank)
lp_vmap = lambda x: jax.vmap(lp, in_axes=0)(x.astype(np.float32))
lp_g_vmap = lambda x: jax.vmap(lp_g, in_axes=0)(x.astype(np.float32))
lp_vmap(ref_samples)
np.save(f"{path}/mean", mean)
np.save(f"{path}/cov", cov)

ranklr = rank
regf = lambda x: args.reg/(1+x)
key = jax.random.PRNGKey(2)
alg = ADVI_LR(D, ranklr, lp_vmap, lp_g_vmap)
#alg = PBAM2(D, lp_vmap, lp_g_vmap)

monitor = KLMonitor(batch_size=32, ref_samples=ref_samples,
                    checkpoint=args.checkpoint, store_params_iter=args.store_params_iter,
                    savepoint=args.savepoint,
                    plot_samples=True,
                    savepath=f'{savepath}/')

np.random.seed(args.seed)
mean = jnp.zeros(D)
psi = 0.1 + np.random.random(D)*0.01
llambda = np.random.normal(0, 0.1, size=(D, ranklr)) 
opt = optax.adam(learning_rate=args.lr)
meanfit, psi, llambda, losses = alg.fit(key, opt,
                                        mean=mean, psi=psi, llambda=llambda,
                                        batch_size=args.batch, niter=args.niter, 
                                        nprint=args.nprint, monitor=monitor)


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

np.save(f'{savepath}/means', monitor.means)
np.save(f'{savepath}/llambdas', monitor.llambdas)
np.save(f'{savepath}/psis', monitor.psis)

#
eps = np.random.normal(0, 1, (2000, D))
z = np.random.normal(0, 1, (2000, ranklr))
s3 = meanfit + psi*eps + (llambda@z.T).T
s = ref_samples[:5000, :] 

dplot = min(D, 5)
fig, ax = plt.subplots(dplot, dplot, figsize=(dplot*1.5, dplot*1.5), sharex='col')
idx = np.random.permutation(np.arange(D))[:dplot]
print(idx)
for i in range(dplot):
    for j in range(dplot):
        ii, jj = idx[i], idx[j]
        if i == j: 
            ax[i, i].hist(s[..., ii], alpha=0.7, density=True, bins=30, color='k', histtype='step', lw=2);
            ax[i, i].hist(s3[..., ii], alpha=0.7, density=True, bins=30, label=f"rank={ranklr}");
            
        elif j > i:
            ax[j, i].plot(s[..., ii], s[..., jj], '.', alpha=1., ms=2, color='k')
            ax[j, i].plot(s3[..., ii], s3[..., jj], '.', alpha=0.5, ms=2)
        else: 
            ax[j, i].set_axis_off()
            
ax[0, 0].legend()
for axis in ax.flatten():
    axis.set_yticks([])
plt.tight_layout()
plt.savefig(f'{savepath}/corner.png')
plt.close()

