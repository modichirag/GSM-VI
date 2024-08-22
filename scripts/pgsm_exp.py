import sys, os
import os

from gsmvi.bam import BAM
from gsmvi.pgsm import PGSM
from gsmvi.monitors import KLMonitor

import matplotlib.pyplot as plt
from imports import *
jax.config.update('jax_platform_name', 'cpu')
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-D', type=int, help='dimension')
parser.add_argument('-rank', type=int, default=32, help='dimension')
parser.add_argument('-noise', type=float, default=0.01, help='rank')
parser.add_argument('--dataseed', type=int, default=0, help='seed between 0-999, default=0')
#arguments for GSM
parser.add_argument('--seed', type=int, default=99, help='seed between 0-999, default=99')
parser.add_argument('--niter', type=int, default=1001, help='number of iterations in training')
parser.add_argument('--batch', type=int, default=2, help='batch size, default=2')
parser.add_argument('--eta', type=float, default=1.2, help='regularizer for ngd and lsgsm')
parser.add_argument('--niter_em', type=int, default=101, help='which regularizer to use')
parser.add_argument('--nprint', type=int, default=10, help='number of times to print')
parser.add_argument('--tolerance', type=float, default=1e-4, help='regularizer for ngd and lsgsm')
#arguments for path name
parser.add_argument('--suffix', type=str, default="", help='suffix, default=""')

print()
args = parser.parse_args()
if args.suffix != '': suffix = f"-{args.suffix}"
else: suffix = ''
D = args.D
rank = args.rank
print(D)

def setup_model(D=10, rank=4):

    # setup a Gaussian target distribution                                                                                                                    
    mean = np.random.random(D)
    L = np.random.normal(size = D*rank).reshape(D, rank)
    cov = np.matmul(L, L.T) + np.diag(np.random.normal(1, 1, D)*1e-1+1)
    model = dist.MultivariateNormal(loc=mean, covariance_matrix=cov)
    lp = jit(lambda x: jnp.sum(model.log_prob(x)))
    lp_g = jit(grad(lp, argnums=0))
    ref_samples = np.random.multivariate_normal(mean, cov, 10000)
    return model, mean, cov, lp, lp_g, ref_samples


#mean, cov, lp, lp_g, ref_samples = setup_gauss_model(D, rank)
model, mean, cov, lp, lp_g, ref_samples = setup_model(D, rank)
lp_vmap = lambda x: jax.vmap(lp, in_axes=0)(x.astype(np.float32))
lp_g_vmap = lambda x: jax.vmap(lp_g, in_axes=0)(x.astype(np.float32))
lp_vmap(ref_samples)

ranklr = rank
key = jax.random.PRNGKey(2)
alg3 = PGSM(D, lp_vmap, lp_g_vmap)
monitor = KLMonitor(batch_size=32, ref_samples=ref_samples, checkpoint=10)
meanfit3, psi, llambda = alg3.fit(key, rank=ranklr,
                                  batch_size=args.batch, niter=args.niter, 
                                  tolerance=args.tolerance, eta=args.eta, niter_em=args.niter_em,
                                  nprint=args.nprint, print_convergence=False, monitor=monitor)



plt.figure(figsize=(7, 3))
plt.subplot(121)
plt.plot(monitor.nevals, np.abs(monitor.rkl))
plt.loglog()
plt.ylabel('reverse kl')
plt.subplot(122)
plt.plot(monitor.nevals, np.abs(monitor.fkl))
plt.loglog()
plt.ylabel('forward kl')
plt.savefig(f'./tmp/pgsm{D}-loss{suffix}.png')
plt.close()


eps = np.random.normal(0, 1, (2000, D))
z = np.random.normal(0, 1, (2000, ranklr))
s3 = meanfit3 + psi*eps + (llambda@z.T).T
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
plt.savefig(f'./tmp/pgsm{D}-hist{suffix}.png')
plt.close()



