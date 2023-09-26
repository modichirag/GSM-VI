###Fits variational distribution with GSM and BBVI for ill-condition Gaussian and saves divergences with iteration
###Corresponds to Fig. 3 of the submitted paper.
import numpy as np
import sys, os, time
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

#
sys.path.append('../src/')
import bbvi, gsm
import gaussianq
import diagnostics as dg
from runs import run_gsm, run_bbvi, make_plot
#import divergences as divs
#import folder_path
#

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-D', type=int, default=4, help='dimension')
parser.add_argument('-c', type=int, default=10, help='condition number')
parser.add_argument('-noise', type=float, default=0.01, help='noise added to diagonal')
#arguments for GSM
parser.add_argument('--dataseed', type=int, default=100, help='seed to generate the target distribution')
parser.add_argument('--seed', type=int, default=0, help='seed between 0-999 to change initialization of variational distribution')
parser.add_argument('--niter', type=int, default=1001, help='number of iterations in training')
parser.add_argument('--batch', type=int, default=2, help='batch size, default=2')
parser.add_argument('--lr', type=float, default=0.01, help='lr for bbvi')
#Arguments for qdist
parser.add_argument('--scale', type=float, default=1., help='scale of Gaussian to initialize')

##################################################################
# Setup the target distribution

print()
args = parser.parse_args()
noise = args.noise
D = args.D
r = args.D
c = args.c

###
# Setup the true target Gaussian distribution as diagonal + rank 'r' covariance matrix
np.random.seed(args.dataseed)
mu = np.random.random(D).astype(np.float32)
L = np.random.normal(size = D*r).reshape(D, r)
cov = tf.constant(np.matmul(L, L.T), dtype=tf.float32) + np.eye(D).astype(np.float32)*noise

#set condition number
eig, eigv = np.linalg.eig(cov)
print(eig)
idx = np.argsort(eig)[::-1]
eig = eig[np.argsort(eig)[::-1]]
eigv = eigv[:, np.argsort(eig)[::-1]]
print(eig)
assert (np.diff(eig[::-1])  > 0 ).all()
assert np.argmax(eig) == 0
assert np.argmin(eig) == D-1
eig = eig/eig[-1] * 0.1 #set min eigenvalue to 0.1
eig[0] *= args.c
cov = tf.constant(np.matmul(eigv, np.matmul(np.diag(eig), np.linalg.inv(eigv))).astype(np.float32))

#and setup model with it
model = gaussianq.FR_Gaussian_cov(d=D, mu=mu, dtype=tf.float32)
model.cov = cov
samples = tf.constant(model.sample(1000))
idx = list(np.arange(D)[:int(min(D, 10))])

##################################################################
##### Setup auxillaries for the runs
#####
# Callback function to monitor training
def callback(qdist, ls, i, savepath, suptitle):
    dg.plot_bbvi_losses(*ls, savepath=savepath, savename='losses_%04d'%i, suptitle=suptitle)
    qsamples = qdist.sample(1000).numpy()
    dg.compare_hist(qsamples[:, idx], samples.numpy()[:, idx], savepath=savepath, savename='hist_%04d'%i, suptitle=suptitle)

#####
# Setup path to save output, save output of true distribution
modelname = f'd{D}-c{c}-n{noise:0.2e}'
modelpath = './tmp/FRG/%s/'%modelname
print(f"\n##### Output will be saved in folder {modelpath} #####\n")

##### 
# Loc esimtate for initialization of variational distribution
np.random.seed(args.seed)
x0 = np.random.random(D).reshape(1, D).astype(np.float32)
x0 = tf.constant(x0)


# ##################################################################
print("\n##### RUN GSM #####\n")
gsmrun = run_gsm(args, model, x0, modelpath, callback, samples)
print("\n##### RUN BBVI #####\n")
bbvirun = run_bbvi(args, model, x0, modelpath, callback, samples)
print("Make diagnostic figure")
make_plot(args, samples, gsmrun, bbvirun, modelpath)
