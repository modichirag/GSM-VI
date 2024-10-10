import sys, os
os.environ['JAX_PLATFORMS'] = 'cpu'

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
parser.add_argument('--rank', type=int, default=32, help='dimension')
parser.add_argument('--noise', type=float, default=0.01, help='rank')
parser.add_argument('--dataseed', type=int, default=0, help='seed between 0-999, default=0')
#arguments for GSM
parser.add_argument('--ranklr', type=int, default=0, help='rank of variational family')
parser.add_argument('--seed', type=int, default=99, help='seed between 0-999, default=99')
parser.add_argument('--niter', type=int, default=1001, help='number of iterations in training')
parser.add_argument('--batch', type=int, default=2, help='batch size, default=2')
parser.add_argument('--nprint', type=int, default=100, help='number of times to print')
parser.add_argument('--lr', type=float, default=1e-2, help='regularizer for ngd and lsgsm')
parser.add_argument('--schedule', type=str, default="", help='scheduler for learning rate')
#args for monitor
parser.add_argument('--checkpoint', type=int, default=10, help='number of times to print')
parser.add_argument('--store_params_iter', type=int, default=50, help='number of times to print')
parser.add_argument('--savepoint', type=int, default=100, help='number of times to print')

#arguments for path name
parser.add_argument('--suffix', type=str, default="", help='suffix, default=""')
parser.add_argument('--cond', type=int, default=0, help='suffix, default=""')

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
if args.cond == 0:  path = f"{basepath}/Gauss-D{D}/R{rank}-seed{args.dataseed}/"
elif args.cond == 1: path = f"{basepath}/Gauss-D{D}-badcond/R{rank}-seed{args.dataseed}/"
elif args.cond == 2:  path = f"{basepath}/Gauss-D{D}-goodcond/R{rank}-seed{args.dataseed}/"
elif args.cond == 3:  path = f"{basepath}/Gauss-D{D}-dnorm/R{rank}-seed{args.dataseed}/"
elif args.cond == 4:  path = f"{basepath}/Gauss-D{D}-ranknorm/R{rank}-seed{args.dataseed}/"
elif args.cond == 5:  path = f"{basepath}/Gauss-D{D}-nonorm/R{rank}-seed{args.dataseed}/"

if args.ranklr != 0:
    ranklr = args.ranklr
    path = path + f'ranklr{args.ranklr}/'
else: ranklr = rank

os.makedirs(path, exist_ok=True)
if args.schedule == "":
    savepath = f'{path}/padvi/B{args.batch}-lr{args.lr:0.3f}{suffix}/'
    schedule = args.lr
elif args.schedule == "linear":
    savepath = f'{path}/padvi/B{args.batch}-lr{args.lr:0.3f}-linschedule{suffix}/'
    schedule = optax.schedules.linear_schedule(args.lr, end_value=1e-5, transition_steps=args.niter)
elif args.schedule == "cosine":
    savepath = f'{path}/padvi/B{args.batch}-lr{args.lr:0.3f}-cosineschedule{suffix}/'
    schedule = optax.schedules.cosine_decay_schedule(args.lr, alpha=1e-5/args.lr, decay_steps=args.niter)

os.makedirs(savepath, exist_ok=True)
print(f"Save results in {savepath}")


if args.cond == 1: mean, cov, lp, lp_g, ref_samples = setup_gauss_model(D, rank) # Bad conditioning
elif args.cond == 2: model, mean, cov, lp, lp_g, ref_samples = setup_model_goodcond(D, rank) # Good conditioning
elif args.cond == 3: model, mean, cov, lp, lp_g, ref_samples = setup_model_dnorm(D, rank)
elif args.cond == 4: model, mean, cov, lp, lp_g, ref_samples = setup_model_ranknorm(D, rank)
elif args.cond == 5: model, mean, cov, lp, lp_g, ref_samples = setup_model_nonorm(D, rank)
elif args.cond == 0: model, mean, cov, lp, lp_g, ref_samples = setup_model(D, rank)

lp_vmap = lambda x: jax.vmap(lp, in_axes=0)(x.astype(np.float32))
lp_g_vmap = lambda x: jax.vmap(lp_g, in_axes=0)(x.astype(np.float32))
lp_vmap(ref_samples)
np.save(f"{path}/mean", mean)
np.save(f"{path}/cov", cov)

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
psi = np.random.random(D)
llambda = np.random.normal(0, 1, size=(D, ranklr))
opt = optax.adam(learning_rate=schedule)
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

