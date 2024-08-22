import jax
import jax.numpy as jnp
from jax import jit, random
from jax.scipy.linalg import sqrtm as sqrtm_jsp
from scipy.linalg import sqrtm as sqrtm_sp
#from numpyro.distributions import MultivariateNormal  ##Needed if sampling from numpyro dist below
import numpy as np
import scipy.sparse as spys
from jax.lib import xla_bridge
from gsmvi.low_rank_gaussian import logp_lr, det_cov_lr, get_diag, monitor_lr
from functools import partial
from time import time



@jit
def low_rank_kl(psi1, llambda1, psi0, alpha, beta):
    K = llambda1.shape[1]

    ltpsinv = llambda1.T*(1/psi1)
    m = jnp.identity(K) + ltpsinv@llambda1
    minv = jnp.linalg.pinv(m)
    mltpsinv = minv@ltpsinv
    det_term = jnp.log(jnp.linalg.det(m)*jnp.prod(psi1))

    t0 = (psi0 + get_diag(alpha, alpha) - get_diag(beta, beta))/psi1
    t1 = psi0*get_diag(ltpsinv.T, mltpsinv.T)
    t2 = get_diag(alpha@(alpha.T@ltpsinv.T), mltpsinv.T)
    t3 = get_diag(beta@(beta.T@ltpsinv.T), mltpsinv.T)
    diag = (t0 - t1 - t2 + t3)
    trace_term = diag.sum()

    kl = trace_term + det_term
    return kl


@jit
def _update_psi_llambda(psi, llambda, alpha, beta, psi0, first_term_1, jitter=1e-6):


    print('jit psi/llambda update')
    # first update llambda
    D, K = llambda.shape
    Id_K = jnp.identity(K)
    
    psi_inv = psi**-1
    C = jnp.linalg.pinv(Id_K + (llambda.T *psi_inv)@llambda) #KxK
    J = C@(llambda.T*psi_inv) #KxD
    SJT = (psi0*J).T + alpha @ (alpha.T@J.T) - beta@(beta.T@J.T)
    llambda_update = SJT @ jnp.linalg.pinv(C + J@SJT)

    # psi update here
    first_term_2 = -2* get_diag(SJT, llambda_update) 

    Mint = J@SJT
    def contractM(a):
        return a@Mint@a
    first_term_3 = jax.vmap(contractM, in_axes=[0])(llambda_update) 
    first_term = first_term_1 + first_term_2 + first_term_3
    
    lupdateC = llambda_update @C 
    second_term = get_diag(lupdateC, llambda_update) 
    psi_update = first_term + second_term + jitter

    return psi_update, llambda_update



def update_psi_llambda(psi, llambda, alpha, beta, niter_em=10, tolerance=0., jit_compile=True, verbose=False, eta=1., min_iter=3):

    assert 1. <= eta < 2
    psi0 = psi.copy()     
    first_term_1 = psi0 + get_diag(alpha, alpha) - get_diag(beta, beta) #diag of cov for psi    
    _update =  _update_psi_llambda
    
    counter = 0
    current_kl = low_rank_kl(psi, llambda, psi0, alpha, beta)
    kld = []
    rkld = []
    for counter in range(niter_em):
        psi_update, llambda_update = _update(psi, llambda, alpha, beta, psi0, first_term_1)

        if np.isnan(psi_update).any() or np.isnan(llambda_update).any():
            return psi, llambda, counter

        else:
            if tolerance == 0:
                psi = (1 - eta)*psi + eta*psi_update
                llambda = (1 - eta)*llambda + eta*llambda_update

            else:
                #use kl to determine if convergeda
                old_kl = current_kl
                current_kl = low_rank_kl(psi_update, llambda_update, psi0, alpha, beta)
                #if np.isnan(current_kl) or np.isinf(current_kl):
                #    print(f'NaN/Inf in KL \n')
                #    #return psi, llambda, counter
                #    return psi*np.NaN, llambda*np.NaN, counter
                
                dist = ((psi_update - psi)**2).sum()**0.5
                psi = (1 - eta)*psi + eta*psi_update
                llambda = (1 - eta)*llambda + eta*llambda_update
                
                rkld.append(np.abs(current_kl/old_kl-1))
                kld.append(current_kl-old_kl)
                #kld.append(current_kl)
                if counter > min_iter:
                    second_der = (kld[-1] + kld[-3] - 2*kld[-2])/kld[-2]
                    if old_kl and (np.array(rkld[-min_iter:]) < tolerance).all() and second_der < tolerance**0.5:
                        norm = ((psi_update)**2).sum()**0.5
                        klds = [f"{i:0.2e}" for i in kld]
                        if verbose: print(f'{counter} iterations to reach convergence. Second derivative {second_der}\n')
                        return psi, llambda, counter
    #print(f'Max EM update iterations reached, {rkld[-min_iter:]}, {second_der}')
    return psi, llambda, counter



#@jit
def gsm_update_single_lr(sample, v, mu0, psi, llambda):
    print('jit gsm update')
    ##############

    D, K = llambda.shape
    S0v = psi*v + llambda@(llambda.T@v)
    vSv = v@S0v
    
    mu_v = jnp.matmul((mu0 - sample), v)
    rho = 0.5 * jnp.sqrt(1 + 4*(vSv + mu_v**2)) - 0.5
    eps0 = S0v - mu0 + sample

    #mu update
    mu_vT = jnp.outer((mu0 - sample), v)
    den = 1 + rho + mu_v
    I = jnp.eye(sample.shape[0])
    mu_update = 1/(1 + rho) * (eps0 - 1/den * (mu0 - sample) * (v@eps0))
    mu = mu0 + mu_update

    #S update
    Supdate_0 =  (mu0-sample)
    Supdate_1 =  (mu-sample)

    return mu_update, Supdate_0, Supdate_1



@jit
def gsm_update_lr(samples, vs, mu0, psi0, llambda0):
    """
    """
    assert len(samples.shape) == 2
    assert len(vs.shape) == 2

    vgsm_update = jax.vmap(gsm_update_single_lr, in_axes=(0, 0, None, None, None))
    mu_update, Supdate_0, Supdate_1 = vgsm_update(samples, vs, mu0, psi0, llambda0)
    mu_update = jnp.mean(mu_update, axis=0)
    mu = mu0 + mu_update
    alpha = jnp.concatenate([llambda0, Supdate_0.T], axis=1)
    beta = Supdate_1.T
    

    #S_update = jnp.mean(S_update, axis=0)
    #S = S0 + S_update

    return mu, alpha, beta



    
class PGSM:
    """
    Wrapper class for using GSM updates to fit a distribution
    """
    def __init__(self, D, lp, lp_g, use_lowrank=False):
        """
        Inputs:
          D: (int) Dimensionality (number) of parameters
          lp : Function to evaluate target log-probability distribution.
               (Only used in monitor, not for fitting)
          lp_g : Function to evaluate score, i.e. the gradient of the target log-probability distribution
        """
        self.D = D
        self.lp = lp
        self.lp_g = lp_g
        self.nan_update = []
        


    def fit(self, key, rank, mean=None, psi=None, llambda=None,
            batch_size=2, niter=5000, nprint=10, niter_em=10, jit_compile=True,
            tolerance=0, print_convergence=False, eta=1.,
            verbose=True, check_goodness=True, monitor=None, retries=10, jitter=1e-6):
        """
        Main function to fit a multivariate Gaussian distribution to the target

        Inputs:
          key: Random number generator key (jax.random.PRNGKey)
          mean : Function to return regularizer value at an iteration. See Regularizers class below
          mean : Optional, initial value of the mean. Expected None or array of size D
          cov : Optional, initial value of the covariance matrix. Expected None or array of size DxD
          batch_size : Optional, int. Number of samples to match scores for at every iteration
          niter : Optional, int. Total number of iterations
          nprint : Optional, int. Number of iterations after which to print logs
          verbose : Optional, bool. If true, print number of iterations after nprint
          check_goodness : Optional, bool. Recommended. Wether to check floating point errors in covariance matrix update
          monitor : Optional. Function to monitor the progress and track different statistics for diagnostics.
                    Function call should take the input tuple (iteration number, [mean, cov], lp, key, number of grad evals).
                    Example of monitor class is provided in utils/monitors.py

        Returns:
          mu : Array of shape D, fit of the mean
          cov : Array of shape DxD, fit of the covariance matrix
        """

        K = rank
        if mean is None:
            mean = jnp.zeros(self.D)

        if llambda is None:
            llambda = np.random.normal(0, 1, size=(self.D, K))*0.1
        if psi is None:
            psi = np.random.random(self.D)

        nevals = 1
        nprojects = []
        update_function = gsm_update_lr
        project_function = partial(update_psi_llambda, jit_compile=jit_compile)
        if jit_compile:
            update_function = jit(update_function)
        if (nprint > niter) and verbose: nprint = niter

        # start loop
        start = time()
        for i in range(niter + 1):
            if (i%(niter//nprint) == 0) and verbose :
                print(f'Iteration {i} of {niter}. Time taken : ', time() - start)
            if i == 1: start = time() # get rid of compile tim
            if i == 10: print("time for first 10 iterations : ", time()-start)
            if monitor is not None:
                if (i%monitor.checkpoint) == 0:
                    monitor_lr(monitor, i, [mean, psi, llambda], self.lp, key, nevals=nevals)
                    nevals = 0

            # Can generate samples from jax distribution (commented below), but using numpy is faster
            j = 0
            while True:         # Sometimes run crashes due to a bad sample. Avoid that by re-trying.
                try:
                    key, key_sample = random.split(key, 2)
                    np.random.seed(key_sample[0])

                    eps = np.random.normal(0, 1, size=(batch_size, self.D))
                    z = np.random.normal(0, 1, size=(batch_size, K))
                    samples = mean + psi**0.5 * eps + (llambda@z.T).T
                    vs = self.lp_g(samples)
                    nevals += batch_size
                    mean_new, alpha, beta = update_function(samples, vs, mean, psi, llambda) # gsm
                    psi_new, llambda_new, counter = project_function(psi, llambda, alpha, beta, \
                                                            niter_em=niter_em, tolerance=tolerance, eta=eta,
                                                            verbose=print_convergence) # project
                    if i == 0: print('compiled')
                    nprojects.append(counter)
                    psi_new +=  jitter # jitter diagonal

                    break
                except Exception as e:
                    if j < retries :
                        j += 1
                        print(f"Failed with exception {e}")
                        print(f"Trying again {j} of {retries}")
                    else : raise e

            is_good = self._check_goodness(mean_new, psi_new, llambda_new)
            if is_good == -1:
                print("Max bad updates reached")
                return mean, psi, llambda
            elif is_good == 1:
                mean, psi, llambda = mean_new, psi_new, llambda_new
            else:
                if verbose: print("Bad update for covariance matrix. Revert")

        if monitor is not None:
            monitor_lr(monitor, i, [mean, psi, llambda], self.lp, key, nevals=nevals)
            monitor.nprojects = nprojects
        print('Total number of projections : ', np.sum(nprojects))
        return mean, psi, llambda


    def _check_goodness(self, mean, psi, cov):
        is_good = 0
        j = 0 
        try:
            for m in [mean, psi, cov]:
                if (np.isnan(m)).any():
                    self.nan_update.append(j)
                    if len(self.nan_update) > 100:
                        is_good = -1
                    return is_good
            is_good = 1
            return is_good
        except:
            return is_good
        

