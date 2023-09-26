###Fits variational distribution with GSM and BBVI for non-Gaussain distribution (Sinh-ArcSinh transformation) and saves divergences with iteration
###Corresponds to Fig. 4 of the submitted paper.
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
import gaussianq
import diagnostics as dg
from runs import run_gsm, run_bbvi, make_plot
#

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-D', type=int, help='dimension')
parser.add_argument('--skewness', type=float, default=0.0, help='rank')
parser.add_argument('--tailw', type=float, default=1.0, help='rank')
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
D = args.D

np.random.seed(args.dataseed)
loc = np.random.random(D).astype(np.float32)
scale = np.random.uniform(0.5, 2., D).astype(np.float32)
skewness = args.skewness
tailweight = args.tailw
basemodel = gaussianq.FR_Gaussian_cov(d=D, mu=loc, scale=scale, dtype=tf.float32)
model = gaussianq.SinhArcsinhTransformation(d=D, loc=0., scale=1.,
                                            skewness=skewness, tailweight=tailweight,
                                            distribution=basemodel.q,
                                            dtype=tf.float32)
samples = tf.constant(model.sample(1000))
print('samples shape : ', samples.shape)
idx = list(np.arange(D)[:int(min(D, 10))])


##################################################################
##### Setup auxillaries for the runs
#####
def callback(qdist, ls, i, savepath, suptitle):
    dg.plot_bbvi_losses(*ls, savepath=savepath, savename='losses_%04d.png'%i, suptitle=suptitle)
    qsamples = qdist.sample(1000).numpy()
    dg.compare_hist(qsamples[:, idx], samples.numpy()[:, idx], savepath=savepath, savename='hist_%04d'%i, suptitle=suptitle)

#####
# Loc esimtate for initialization of variational distribution
np.random.seed(args.seed)
x0 = np.random.random(D).reshape(1, D).astype(np.float32)
x0 = tf.constant(x0)

#####
# Setup path to save output, save output of true distribution
modelname = f'd{D}-s{skewness:0.1f}-t{tailweight:0.1f}'
modelpath = './tmp//ArchSinh/%s/'%modelname
print(f"\n##### Output will be saved in folder {modelpath} #####\n")


##################################################################

print("\n##### RUN GSM #####\n")
gsmrun = run_gsm(args, model, x0, modelpath, callback, samples)
print("\n##### RUN BBVI #####\n")
bbvirun = run_bbvi(args, model, x0, modelpath, callback, samples)
print("Make diagnostic figure")
make_plot(args, samples, gsmrun, bbvirun, modelpath)
