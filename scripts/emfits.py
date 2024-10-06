import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

# enable 16 bit precision for jax                                                                                                                             
from jax import config
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from jax import jit, grad, random
# Import GSM                                                                                                                                                  
import sys
sys.path.append('../gsmvi/')
from em_lr_projection import fit_lr_gaussian_law, fit_lr_gaussian_diana
#####                                                                                                                                                         

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-D', type=int, help='dimension')
parser.add_argument('-K', type=int, default=32, help='dimension')
parser.add_argument('--noise', type=float, default=1., help='rank')
parser.add_argument('--dataseed', type=int, default=0, help='seed between 0-999, default=0')
#arguments for GSM
parser.add_argument('--seed', type=int, default=99, help='seed between 0-999, default=99')
parser.add_argument('--niter', type=int, default=101, help='number of iterations in training')
parser.add_argument('--tolerance', type=float, default=1e-4, help='regularizer for ngd and lsgsm')
parser.add_argument('--initname', type=str, default="random", help='suffix, default=""')
parser.add_argument('--lscale', type=float, default=1., help='regularizer for ngd and lsgsm')
parser.add_argument('--pscale', type=float, default=1., help='regularizer for ngd and lsgsm')
#arguments for path name
parser.add_argument('--suffix', type=str, default="", help='suffix, default=""')

print()
args = parser.parse_args()

D = args.D
K = args.K
tol = args.tolerance
eta = 1.2
niter = args.niter
num_of_latents = K

#gen data
dataseed = args.dataseed
np.random.seed(dataseed)
noise = args.noise
psi_true = np.diag(np.random.random(D))*noise
L = np.random.normal(size = D*K).reshape(D, K) / (D*K)**0.5
llambda_true = L
cov = np.matmul(L, L.T) + psi_true
eigs = np.linalg.eigvals(cov)
print("condition number : ", np.linalg.cond(cov))

savedir = f'/mnt/ceph/users/cmodi/pbam/emfits/D{D}_K{K}_S{dataseed}_N{noise:0.2f}/'
print("save runs in : ", savedir)
os.makedirs(savedir, exist_ok=True)
np.save(f'{savedir}/psi', psi_true)
np.save(f'{savedir}/llambda', llambda_true)

#initialize
subsavedir = f'{savedir}/{args.initname}_S{args.seed}_P{args.pscale}_L{args.lscale}/'
os.makedirs(subsavedir, exist_ok=True)
print("save config in : ", subsavedir)
fits = {'law':{}, 'diana':{}}

if args.initname == 'llambda0':
    np.random.seed(args.seed)
    psi_init, llambda_init = np.diag(np.random.random(D)),  np.random.normal(size=(D,K))*args.lscale
    
    psi_temp = fit_lr_gaussian_law(cov, num_of_latents=K, 
                                   num_of_itr=niter, diagnosis=True, tolerance=tol, eta=eta, 
                                   psi=psi_init.copy(), llambda=llambda_init*0.)[0]
    fits['law']['psi'], fits['law']['llambda'], fits['law']['losses']  = fit_lr_gaussian_law(cov, num_of_latents, 
                                                                                             num_of_itr=niter, diagnosis=True, tolerance=tol, eta=eta, 
                                                                                             psi=psi_temp.copy(), llambda=llambda_init.copy())
    #cov_law = fits['law']['psi'] + fits['law']['llambda']@fits['law']['llambda'].T
    for key in ['law']:
        os.makedirs(f"{subsavedir}/{key}/", exist_ok=True)
        np.save(f'{subsavedir}/{key}/psi', fits[key]['psi'])
        np.save(f'{subsavedir}/{key}/llambda', fits[key]['llambda'])
        np.save(f'{subsavedir}/{key}/psis', fits[key]['losses'][0])
        np.save(f'{subsavedir}/{key}/llambdas', fits[key]['losses'][1])
        np.save(f'{subsavedir}/{key}/kls', fits[key]['losses'][2])
    
    psi_temp = fit_lr_gaussian_diana(cov, num_of_latents=K, 
                                     num_of_itr=niter, diagnosis=True, tolerance=tol, eta=eta, 
                                     psi=psi_init, llambda=llambda_init*0)[0]
    fits['diana']['psi'], fits['diana']['llambda'], fits['diana']['losses'] = fit_lr_gaussian_diana(cov, num_of_latents, 
                                                                                            num_of_itr=niter, diagnosis=True, tolerance=tol, eta=eta, 
                                                                                                    psi=psi_temp.copy(), llambda=llambda_init.copy())
    #cov_diana = fits['law']['psi'] + fits['law']['llambda']@fits['law']['llambda'].T

    
else:
    if args.initname == 'random':
        np.random.seed(args.seed)
        psi_init, llambda_init = np.diag(np.random.random(D))*args.pscale,  np.random.normal(size=(D,K))*args.lscale
    if args.initname == 'psi_id':
        np.random.seed(args.seed)
        psi_init, llambda_init = np.eye(D),  np.random.normal(size=(D,K))*args.lscale

    fits['law']['psi'], fits['law']['llambda'], fits['law']['losses']  = fit_lr_gaussian_law(cov, num_of_latents=K, 
                                                    num_of_itr=niter, diagnosis=True, tolerance=tol, eta=eta, 
                                                   psi=psi_init.copy(), llambda=llambda_init.copy())
    #cov_law = fits['law']['psi'] + fits['law']['llambda']@fits['law']['llambda'].T
    for key in ['law']:
        os.makedirs(f"{subsavedir}/{key}/", exist_ok=True)
        np.save(f'{subsavedir}/{key}/psi', fits[key]['psi'])
        np.save(f'{subsavedir}/{key}/llambda', fits[key]['llambda'])
        np.save(f'{subsavedir}/{key}/psis', fits[key]['losses'][0])
        np.save(f'{subsavedir}/{key}/llambdas', fits[key]['losses'][1])
        np.save(f'{subsavedir}/{key}/kls', fits[key]['losses'][2])


    fits['diana']['psi'], fits['diana']['llambda'], fits['diana']['losses'] = fit_lr_gaussian_diana(cov, num_of_latents=K, 
                                                    num_of_itr=niter, diagnosis=True, tolerance=tol, eta=eta, 
                                                   psi=psi_init, llambda=llambda_init)
    #cov_diana = fits['law']['psi'] + fits['law']['llambda']@fits['law']['llambda'].T



for key in ['law', 'diana']:
    os.makedirs(f"{subsavedir}/{key}/", exist_ok=True)
    np.save(f'{subsavedir}/{key}/psi', fits[key]['psi'])
    np.save(f'{subsavedir}/{key}/llambda', fits[key]['llambda'])
    np.save(f'{subsavedir}/{key}/psis', fits[key]['losses'][0])
    np.save(f'{subsavedir}/{key}/llambdas', fits[key]['losses'][1])
    np.save(f'{subsavedir}/{key}/kls', fits[key]['losses'][2])

