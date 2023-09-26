import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
from jax import config
config.update("jax_enable_x64", True)

from imports import *
import tensorflow_probability as tfp
SinhArcsinh = tfp.substrates.jax.distributions.SinhArcsinh

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--alg', type=str, help='which algorithm to run')
parser.add_argument('-D', type=int, default=10, help='dimension')
parser.add_argument('-s', type=float, help='skewness')
parser.add_argument('-t', type=float, help='tail weight')
parser.add_argument('-noise', type=float, default=0.01, help='rank')
parser.add_argument('--dataseed', type=int, default=123, help='seed between 0-999, default=0')
#arguments for GSM
parser.add_argument('--seed', type=int, default=99, help='seed between 0-999, default=99')
parser.add_argument('--niter', type=int, default=1001, help='number of iterations in training')
parser.add_argument('--batch', type=int, default=2, help='batch size, default=2')
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
parser.add_argument('--reg', type=float, default=1e-5, help='regularizer for ngd')
#arguments for path name
parser.add_argument('--suffix', type=str, default="", help='suffix, default=""')

print()
args = parser.parse_args()
D, s, t = args.D, args.s, args.t
lr, batch_size = args.lr, args.batch
alg = args.alg


def setup_sinh_arcsinh_model(D, s, t, seed):

    np.random.seed(seed)
    mean = np.random.random(D)
    L = np.random.normal(size = D**2).reshape(D, D)
    cov = np.matmul(L, L.T) + np.eye(D)*1e-3
    scale = np.diagonal(cov)
    skewness = s #np.random.uniform(0, 1.0, D)
    tailweight = t #np.random.uniform(0.5, 1.5, D)
    model = SinhArcsinh(mean, scale, skewness=skewness, tailweight=tailweight)                        
    lp = jit(lambda x: jnp.sum(model.log_prob(x)))
    lp_g = jit(grad(lp, argnums=0))
    rng = jax.random.PRNGKey(0)
    ref_samples = model.sample(5000, rng)
    return model, mean, scale, lp, lp_g, ref_samples


model, mean, scale, lp, lp_g, ref_samples = setup_sinh_arcsinh_model(D, s, t, seed=args.dataseed)
print("mean : ", mean)
path = f"/mnt/ceph/users/cmodi/polyakVI/gsmpaper_rebuttal/Sinh-Archsinh/D{D}-s{s:0.1f}-t{t:0.1f}/"
os.makedirs(path, exist_ok=True)
np.save(f"{path}/mean", mean)
np.save(f"{path}/scale", scale)
np.save(f"{path}/ref_samples", ref_samples)

##
monitor = Monitor(batch_size=32, ref_samples=ref_samples)
key = random.PRNGKey(99)
print("For algorithm : ", alg)
if alg == 'gsm':
    gsm = GSM(D=D, lp=lp, lp_g=lp_g)
    path = f"{path}/{alg}/B{batch_size}/"    
    mean_fit, cov_fit = gsm.fit(key, batch_size=batch_size,
                                niter=args.niter, monitor=monitor)

elif alg == 'advi':
    advi = ADVI(D=D, lp=lp)
    path = f"{path}/{alg}/B{batch_size}-lr{lr:0.3f}/"    
    opt = optax.adam(learning_rate=lr)
    mean_fit, cov_fit, losses = advi.fit(key, opt, batch_size=batch_size, 
                                         niter=args.niter, monitor=monitor)
elif alg == 'ngd':
    ngd = NGD(D=D, lp=lp, lp_g=lp_g)
    path = f"{path}/{alg}/B{batch_size}-lr{lr:0.3f}/"    
    mean_fit, cov_fit = ngd.fit(key, lr=args.lr, batch_size=batch_size,
                                reg=args.reg, niter=args.niter, monitor=monitor)
    

os.makedirs(path, exist_ok=True)
np.save(f"{path}/mean", mean_fit)
np.save(f"{path}/cov", cov_fit)
np.save(f"{path}/nevals", monitor.nevals)
np.save(f"{path}/bkl", monitor.bkl)
np.save(f"{path}/fkl", monitor.fkl)
    
