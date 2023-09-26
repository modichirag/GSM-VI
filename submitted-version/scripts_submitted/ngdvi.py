###Fits variational distribution with GSM and BBVI for a full-rank Gaussian and saves divergences with iteration
###Corresponds to Fig. 2 of the submitted paper.
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

from gmmvi.gmmvi_runner import GmmviRunner
from gmmvi.configs import get_default_algorithm_config, update_config

#
sys.path.append('../src_tf/')
import gaussianq
import diagnostics as dg
from runs import run_gsm, run_bbvi, make_plot
#

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-D', type=int, default=4, help='dimension')
parser.add_argument('-r', type=int, default=4, help='rank')
parser.add_argument('-noise', type=float, default=0.01, help='noise added to diagonal')
#arguments for GSM
parser.add_argument('--dataseed', type=int, default=100, help='seed to generate the target distribution')
parser.add_argument('--seed', type=int, default=0, help='seed between 0-999 to change initialization of variational distribution')
parser.add_argument('--niter', type=int, default=1001, help='number of iterations in training')
parser.add_argument('--batch', type=int, default=2, help='batch size, default=2')
parser.add_argument('--lr', type=float, default=0.01, help='lr for bbvi')
#Arguments for qdist
parser.add_argument('--scale', type=float, default=1., help='scale of Gaussian to initilize')


def run_gmmvi(batch, model, x0, modelpath):
    '''This function is based on the following example https://gmmvi.readthedocs.io/en/latest/get_started.html#using-the-gmmvirunner-with-custom-environments
    '''

# For creating a custom environment, we need to extend
# gmmvi.experiments.target_distributions.lnpdf.LNPDF:
    from gmmvi.experiments.target_distributions.lnpdf import LNPDF
    class Target(LNPDF):
        def __init__(self):
            super(Target, self).__init__(safe_for_tf_graph=False)
            self.model = model

        def get_num_dimensions(self) -> int:
            return self.model.d

        def log_density(self, samples: tf.Tensor) -> tf.Tensor:
            return self.model.log_prob(samples)


    # We can also use the GmmviRunner, when using custom environments, but we have
    # to put the LNPDF object into the dict. Furthermore, we need to define the other
    # environment-specific settings that would otherwise be defined in
    # the corresponding config in gmmvi/config/experiment_configs:
    environment_config = {
        "target_fn": None, ## I somehow couldn't add Target() here, but I had to add it after merge_configs()
        "start_seed": 0,
        "environment_name": "GSMTARGET",
        "model_initialization": {
            "use_diagonal_covs": False,
            "num_initial_components": 1,
            # Does GSM/BBVI use the same initial Gaussian???
            "prior_mean": 0., 
            "prior_scale": 1.,
            "initial_cov": 1.,
        },
        "gmmvi_runner_config": {
            "log_metrics_interval": 1
        },
        "use_sample_database": True,
        "max_database_size": int(1e6),
        "temperature": 1.,
        "dump_gmm_path" :  f'{modelpath}/NGD-B{args.batch}'

    }

    algorithm_config = get_default_algorithm_config("SEMTRON") # The recommended variant is SAMTRON, SEMTRON does not add additional components
    algorithm_config['sample_selector_config']['desired_samples_per_component']= args.batch # Only hyperparameter worth tuning?

    # Now we just need to merge the configs and use GmmviRunner as before:
    merged_config = update_config(algorithm_config, environment_config)
    merged_config['target_fn']=Target()
    gmmvi_runner = GmmviRunner.build_from_config(merged_config)

    for epoch in range(args.niter):
        print(epoch)
        gmmvi_runner.iterate_and_log(epoch)
        gmmvi_runner.log_to_disk(epoch)
    return None, None, None, None





##################################################################
# Setup the target distribution
print()
args = parser.parse_args()
noise = args.noise
D = args.D
r = args.r

###
# Setup the true target Gaussian distribution as diagonal + rank 'r' covariance matrix
np.random.seed(args.dataseed)
mu = np.random.random(D).astype(np.float32)
L = np.random.normal(size = D*r).reshape(D, r)
cov = tf.constant(np.matmul(L, L.T), dtype=tf.float32) + np.eye(D).astype(np.float32)*noise
model = gaussianq.FR_Gaussian_cov(d=D, mu=mu, dtype=tf.float32)
model.cov = cov
#samples = tf.constant(model.sample(1000))
samples = np.random.multivariate_normal(mu, cov, 5000)
idx = list(np.arange(D)[:int(min(D, 10))])

modelname = f'd{D}-r{r}-n{noise:0.2e}'
modelpath = './tmp//FRG/%s/'%modelname
os.makedirs(modelpath, exist_ok=True)

np.save(f'{modelpath}/mean', mu)
np.save(f'{modelpath}/cov', cov)
np.save(f'{modelpath}/samples', samples)


##################################################################
##### Setup auxillaries for the runs
######

# #####
# # Setup path to save output, save output of true distribution
# print(f"\n##### Output will be saved in folder {modelpath} #####\n")

##### 
# Loc esimtate for initialization of variational distribution
np.random.seed(args.seed)
x0 = np.random.random(D).reshape(1, D).astype(np.float32)
x0 = tf.constant(x0)


##################################################################

print("\n##### RUN NGD #####\n")
gsmrun = run_gmmvi(args, model, x0, modelpath)
#print("\n##### RUN BBVI #####\n")
