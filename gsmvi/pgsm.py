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
    #minv = jnp.linalg.pinv(m)
    #mltpsinv = minv@ltpsinv
    mltpsinv = jnp.linalg.solve(m, ltpsinv)
    det_term =  jnp.linalg.slogdet(m)[1] + jnp.sum(jnp.log(psi1))

    t0 = (psi0 + get_diag(alpha, alpha) - get_diag(beta, beta))/psi1
    t1 = psi0*get_diag(ltpsinv.T, mltpsinv.T)
    t2 = get_diag(alpha@(alpha.T@ltpsinv.T), mltpsinv.T)
    t3 = get_diag(beta@(beta.T@ltpsinv.T), mltpsinv.T)
    diag = (t0 - t1 - t2 + t3)
    trace_term = diag.sum()

    kl = trace_term + det_term
    return kl


@jit
def _update_psi_llambda(psi, llambda, alpha, beta, psi0, first_term_1, jitter=1e-4):


    print('jit psi/llambda update')
    # first update llambda
    D, K = llambda.shape
    Id_K = jnp.identity(K)
    
    psi_inv = psi**-1
    C = jnp.linalg.inv(Id_K + (llambda.T *psi_inv)@llambda) #KxK
    J = C@(llambda.T*psi_inv) #KxD
    SJT = (psi0*J).T + alpha @ (alpha.T@J.T) - beta@(beta.T@J.T)
    #llambda_update = SJT @ jnp.linalg.pinv(C + J@SJT)
    llambda_update = jnp.linalg.solve((C + J@SJT).T, SJT.T).T 

    # psi update here
    first_term_2 = -2* get_diag(SJT, llambda_update) 
    Mint = J@SJT
    def contractM(a):
        return a@Mint@a
    first_term_3 = jax.vmap(contractM, in_axes=[0])(llambda_update) 
    first_term = first_term_1 + first_term_2 + first_term_3
    
    lupdateC = llambda_update @C 
    second_term = get_diag(lupdateC, llambda_update) 
    psi_update = jnp.maximum(first_term + second_term,  jitter)

    return psi_update, llambda_update



def update_psi_llambda(psi, llambda, alpha, beta, niter_em=10, tolerance=0., jit_compile=True, verbose=False, eta=1., min_iter=3, gamma=0.):

    assert 1. <= eta < 2
    if (eta > 1.) and (gamma != 0):
        print("Double accelration. Exit")
        return None
    psi0 = psi.copy()     
    first_term_1 = psi0 + get_diag(alpha, alpha) - get_diag(beta, beta) #diag of cov for psi
    
    counter = 0
    current_kl = low_rank_kl(psi, llambda, psi0, alpha, beta)
    kld = []
    rkld = []
    psi_prev, llambda_prev = 0., 0.
    for counter in range(niter_em):
        psi_update, llambda_update = _update_psi_llambda(psi, llambda, alpha, beta, psi0, first_term_1)

        if np.isnan(psi_update).any() or np.isnan(llambda_update).any():
            print(f'{counter} nan in update')
            return psi*np.NaN, llambda, counter

        else:
            
            if eta > 1:
                psi = (1 - eta)*psi + eta*psi_update
                llambda = (1 - eta)*llambda + eta*llambda_update
            elif (gamma != 0):
                psi_hold, llambda_hold = psi.copy(), llambda.copy()
                psi = psi_update + gamma*(psi - psi_prev)
                llambda  = llambda_update + gamma*(llambda - llambda_prev)
                psi_prev, llambda_prev = psi_hold, llambda_hold
            else:
                psi, llambda = psi_update, llambda_update
                
            if tolerance != 0:

                #use kl to determine if convergeda
                old_kl = current_kl
                current_kl = low_rank_kl(psi_update, llambda_update, psi0, alpha, beta)
                if np.isnan(current_kl) or np.isinf(current_kl):
                   print(f'NaN/Inf in KL \n')
                #    #return psi, llambda, counter
                #    return psi*np.NaN, llambda*np.NaN, counter
                
                dist = ((psi_update - psi)**2).sum()**0.5                
                rkld.append(np.abs(current_kl/old_kl-1))
                kld.append(current_kl-old_kl)
                #kld.append(current_kl)
                if counter > min_iter:
                    second_der = (kld[-1] + kld[-3] - 2*kld[-2])/kld[-2]
                    # if old_kl and (np.array(rkld[-1]) < tolerance).all() :
                    #     if verbose: print(f'{counter} iterations to reach convergence')
                    #     return psi, llambda, counter

                    if old_kl and (np.array(rkld[-min_iter:]) < tolerance).all() and second_der < tolerance**0.5:
                        norm = ((psi_update)**2).sum()**0.5
                        klds = [f"{i:0.2e}" for i in kld]
                        if verbose: print(f'{counter} iterations to reach convergence. Second derivative {second_der}')
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

    return mu, alpha, beta



def simultaneous_power_iteration(u, v, k, Q=None):
    
    # n, m = A.shape
    np.random.seed(1)
    n = u.shape[0]
    if Q is None: Q = np.random.rand(n, k)
    Q, _ = np.linalg.qr(Q)
    Q_prev = Q
    # print(A.shape, Q.shape)
    
    for i in range(100):
        Z = u@(u.T@(Q)) - v@(v.T@(Q))
        Q, R = np.linalg.qr(Z)
        # can use other stopping criteria as well 
        err = ((Q - Q_prev) ** 2).sum()
        if i % 10 == 0:
            pass
            # print(i, err)
        Q_prev = Q
        if err < 1e-3:
            break
    return np.diag(R), Q


    
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
        

    def init_func(self, rank, mean, psi, llambda):
        samples = np.random.normal(0, 1, size=(rank, self.D))
        vs = self.lp_g(samples)    
        mean, alpha, beta = gsm_update_lr(samples, vs, mean, psi, llambda) # gsm
        true_diag = get_diag(alpha, alpha) - get_diag(beta, beta)
        u = alpha[:, rank:]
        v = beta
        llambda = simultaneous_power_iteration(u, v, rank)[1]
        ldiag = get_diag(llambda, llambda)
        psi = true_diag - ldiag
        return mean, psi, llambda
        

    def fit(self, key, rank, mean=None, psi=None, llambda=None,
            batch_size=2, niter=5000, nprint=10, niter_em=10, jit_compile=True,
            tolerance=0, print_convergence=False, eta=1., gamma=0.,
            scalellambda = 0.1,
            verbose=True, check_goodness=True, monitor=None, retries=10, jitter=1e-4):
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
            llambda = np.random.normal(0, 1, size=(self.D, K)) *scalellambda
        if psi is None:
            psi = 1+np.random.random(self.D)

        #mean, psi, llambda = self.init_func(rank, mean, psi, llambda)
        
        nevals = 1
        nprojects = []
        update_function = gsm_update_lr
        project_function = partial(update_psi_llambda, jit_compile=jit_compile)
        if jit_compile:
            update_function = jit(update_function)
        if (nprint > niter) and verbose: nprint = niter
        tol_factor = 1.
        niter_factor = 1.
        best_rkl = np.inf
        
        # start loop
        # start loop
        if monitor is not None:
            monitor.means = []
            monitor.psis = []
            monitor.llambdas = []
            monitor.iparams = []
            monitor.nprojects = []
        start = time()
        mean_prev, psi_prev, llambda_prev = mean, psi, llambda
        for i in range(niter + 1):
            if (i%(niter//nprint) == 0) and verbose :
                print(f'Iteration {i} of {niter}. Time taken : ', time() - start)
            if i == 1: start = time() # get rid of compile tim
            if i == 10: print("time for first 10 iterations : ", time()-start)
            if monitor is not None:
                if (i%monitor.checkpoint) == 0:
                    monitor_lr(monitor, i, [mean, psi, llambda], self.lp, key, nevals=nevals)
                    # keep a copy of the best fit so far
                    if monitor.rkl[-1] < best_rkl:
                        #print('new best kl : (best, current) : ', best_rkl, monitor.rkl[-1])
                        best_rkl = monitor.rkl[-1]
                        mean_prev, psi_prev, llambda_prev = mean, psi, llambda
                    if (monitor.rkl[-1] > 0) & (monitor.rkl[-1] > 2*best_rkl):
                        print('diverging? : (best, current) : ', best_rkl, monitor.rkl[-1])
                        mean, psi, llambda = mean_prev, psi_prev, llambda_prev
                        self.nan_update.append(i)                    
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
                    #if i == 0 : samples = eps
                    #else: samples = mean + psi**0.5 * eps + (llambda@z.T).T
                    vs = self.lp_g(samples)
                    nevals += batch_size
                    mean_new, alpha, beta = update_function(samples, vs, mean, psi, llambda) # gsm
                    psi_new, llambda_new, counter = project_function(psi, llambda, alpha, beta, \
                                                                     niter_em = niter_em,
                                                                     tolerance = tolerance,
                                                                     #niter_em=int(niter_em *niter_factor),
                                                                     #tolerance=tolerance * tol_factor,
                                                                     eta=eta, gamma=gamma,
                                                                     verbose=print_convergence) # project
                    if i == 0: print('compiled')
                    monitor.nprojects.append(counter)
                    psi_new = jnp.maximum(psi_new, jitter)

                    break
                except Exception as e:
                    if j < retries :
                        j += 1
                        print(f"Failed with exception {e}")
                        print(f"Trying again {j} of {retries}")
                    else : raise 

            is_good = self._check_goodness(mean_new, psi_new, llambda_new)
            if is_good == -1:
                print("Max bad updates reached")
                return mean, psi, llambda
            elif is_good == 1:
                tol_factor = 1.
                #mean_prev, psi_prev, llambda_prev = mean*1., psi*1., llambda.copy()
                mean, psi, llambda = mean_new, psi_new, llambda_new
            else:
                #tol_factor /= 10.
                #niter_factor *= 2.
                if mean_prev is not None:
                    mean, psi, llambda = mean_prev, psi_prev, llambda_prev
                #else:
                #    i = 0
                if verbose: print("Bad update for covariance matrix. Revert")

        if monitor is not None:
            monitor_lr(monitor, i, [mean, psi, llambda], self.lp, key, nevals=nevals)
        print('Total number of projections : ', np.sum(monitor.nprojects), np.sum(monitor.nprojects)/i)
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
        

