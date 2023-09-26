import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
from jax import config
config.update("jax_enable_x64", True)

from imports import *

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--alg', type=str, help='which algorithm to run')
parser.add_argument('-D', type=int, help='dimension')
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
D = args.D
lr, batch_size = args.lr, args.batch
alg = args.alg

mean, cov, lp, lp_g, ref_samples = setup_gauss_model(args.D, seed=args.dataseed)
print("mean : ", mean)
path = f"/mnt/ceph/users/cmodi/polyakVI/gsmpaper_rebuttal/FRG/D{D}/"
os.makedirs(path, exist_ok=True)
np.save(f"{path}/mean", mean)
np.save(f"{path}/cov", cov)
    
monitor = Monitor(batch_size=32, ref_samples=ref_samples)
key = random.PRNGKey(99)
print(f"save in : {path}")
print("For algorithm : ", alg)
print("For lr and batch : ", lr, batch_size)

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
    
