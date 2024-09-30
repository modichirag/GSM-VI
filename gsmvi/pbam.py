import jax
import jax.numpy as jnp
from jax import jit, random
from jax.scipy.linalg import sqrtm as sqrtm_jsp
from scipy.linalg import sqrtm as sqrtm_sp
#from numpyro.distributions import MultivariateNormal  ##Needed if sampling from numpyro dist below
import numpy as np
import scipy.sparse as spys
from jax.lib import xla_bridge
from gsmvi.em_lr_projection import fit_lr_gaussian, project_lr_gaussian
from gsmvi.bam import bam_update, bam_lowrank_update, get_sqrt
from gsmvi.low_rank_gaussian import logp_lr, det_cov_lr, get_diag, monitor_lr
from gsmvi import plotting
from functools import partial
from time import time


@jit
def solve_right(A, B):
    return jnp.linalg.solve(B.T, A.T).T


@jit
def low_rank_kl(psi1, llambda1, psi0, R, VTQM, QTV):
    K = llambda1.shape[1]
    ltpsinv = llambda1.T*(1/psi1)
    m = jnp.identity(K) + ltpsinv@llambda1
    #minv = jnp.linalg.pinv(m)
    #mltpsinv = minv@ltpsinv
    mltpsinv = jnp.linalg.solve(m, ltpsinv)
    t0 = (psi0 + get_diag(R, R) - get_diag(VTQM, QTV.T))/psi1
    t1 = psi0*get_diag(ltpsinv.T, mltpsinv.T)
    t2 = get_diag(R@(R.T@ltpsinv.T), mltpsinv.T)
    t3 = get_diag(VTQM@(QTV@ltpsinv.T), mltpsinv.T)
    diag = (t0 - t1 - t2 + t3)
    trace_term = diag.sum()
    #det_term = jnp.log(jnp.linalg.det(m)*jnp.prod(psi1))
    #print(jnp.linalg.slogdet(m)[1])
    det_term =  jnp.linalg.slogdet(m)[1] + jnp.sum(jnp.log(psi1))
    kl = trace_term + det_term
    return kl



@jit
def _update_psi_llambda(psi, llambda, R, VTQM, QTV, psi0, first_term_1, jitter=1e-6):
    print('jit psi/llambda update')
    # first update llambda
    D, K = llambda.shape
    Id_K = jnp.identity(K)
    psi_inv = psi**-1
    C = jnp.linalg.inv(Id_K + (llambda.T *psi_inv)@llambda) #KxK  #replace by Solve?
    J = C@(llambda.T*psi_inv) #KxD
    VJT = (J*psi0).T + R@(R.T@J.T) #DxK
    SJT = VJT - VTQM@(QTV@J.T)   #SigmaJT , DxK
    # llambda_update = SJT @ jnp.linalg.pinv(C + J@SJT)  #replace by Solve?
    llambda_update = jnp.linalg.solve((C + J@SJT).T, SJT.T).T 

    # now update psi.
    first_term_2 = -2* get_diag(SJT, llambda_update)
    Mint = J@SJT
    def contractM(a):
        return a@Mint@a
    first_term_3 = jax.vmap(contractM, in_axes=[0])(llambda_update) 
    first_term = first_term_1 + first_term_2 + first_term_3
    second_term = get_diag(llambda_update @C , llambda_update)
    psi_update = jnp.maximum(first_term + second_term,  jitter)

    return psi_update, llambda_update


@jit
def _update_psi_llambda_diana(psi, llambda, R, VTQM, QTV, psi0, first_term, jitter=1e-6):

    
    D, K = llambda.shape
    Id_K = jnp.identity(K)
    Id_D = jnp.identity(D)

    psi_inv = psi**-1
    alpha = psi_inv*llambda.T
    # beta = alpha @ (Id_D - llambda@jnp.linalg.pinv(Id_K + alpha@llambda) @alpha)    
    beta = alpha @ (Id_D - llambda @ jnp.linalg.solve(Id_K + alpha@llambda, alpha))    
    SBT = (psi0 *beta).T + R@(R.T@beta.T) - VTQM@(QTV@beta.T)    
    # llambda_update = SBT@jnp.linalg.pinv(beta@SBT + Id_K - beta@llambda)
    llambda_update = solve_right(SBT, beta@SBT + Id_K - beta@llambda)                
    second_term = get_diag(llambda_update, SBT)
    psi_update = (first_term - second_term)
    
    return psi_update, llambda_update


def update_psi_llambda(psi, llambda, R, VTQM, QTV, niter_em=10, tolerance=0., jit_compile=True, verbose=False, eta=1., min_iter=3, psi0=None, jitter=1e-6):

    assert 1. <= eta < 2
    if psi0 is None: psi0 = psi.copy()     
    first_term_1 = psi0 + get_diag(R, R) - get_diag(VTQM, QTV.T) #diag of cov for psi
    #if jit_compile :  _update = jit(_update_psi_llambda)
    #else:   _update = _update_psi_llambda
    _update =  _update_psi_llambda
    
    counter = 0
    current_kl = low_rank_kl(psi, llambda, psi0, R, VTQM, QTV)
    kld = []
    rkld = []
    for counter in range(niter_em):
        psi_update, llambda_update = _update(psi, llambda, R, VTQM, QTV, psi0, first_term_1, jitter=jitter)

        if np.isnan(psi_update).any() or np.isnan(llambda_update).any():
            return psi*np.NaN, llambda, counter

        else:
            if tolerance == 0:
                psi = (1 - eta)*psi + eta*psi_update
                llambda = (1 - eta)*llambda + eta*llambda_update

            else:
                #use kl to determine if convergeda
                old_kl = current_kl
                current_kl = low_rank_kl(psi_update, llambda_update, psi0, R, VTQM, QTV)
                if np.isnan(current_kl) or np.isinf(current_kl):
                   print(f'{counter}, NaN/Inf in KL \n')
                   #return psi, llambda, counter
                   return psi*np.NaN, llambda*np.NaN, counter
                
                dist = ((psi_update - psi)**2).sum()**0.5
                psi = (1 - eta)*psi + eta*psi_update
                llambda = (1 - eta)*llambda + eta*llambda_update
                rkld.append(np.abs(current_kl/old_kl-1))
                kld.append(current_kl-old_kl)
                #kld.append(current_kl)
                if counter > min_iter:
                    second_der = (kld[-1] + kld[-3] - 2*kld[-2])/kld[-2]
                    if old_kl and (np.array(rkld[-1]) < tolerance).all() :
                        if verbose: print(f'{counter} iterations to reach convergence')
                        return psi, llambda, counter
                    # second_der = (kld[-1] + kld[-3] - 2*kld[-2])/kld[-2]
                    # if old_kl and (np.array(rkld[-min_iter:]) < tolerance).all() and second_der < tolerance**0.5:
                    #     norm = ((psi_update)**2).sum()**0.5
                    #     klds = [f"{i:0.2e}" for i in kld]
                    #     if verbose: print(f'{counter} iterations to reach convergence. Second derivative {second_der}')
                    #     print(f'{counter} iterations to reach convergence. Second derivative {second_der}')
                    #     return psi, llambda, counter
    #print(f'Max EM update iterations reached, {rkld[-min_iter:]}, {second_der}')
    return psi, llambda, counter



@jit
def pbam_update(samples, vs, mu0, psi0, llambda0, reg):
    print('jit pbam update')
    ##############
    assert len(samples.shape) == 2
    assert len(vs.shape) == 2
    B, D = samples.shape
    K = llambda0.shape[1]
    xbar = jnp.mean(samples, axis=0)
    gbar = jnp.mean(vs, axis=0)
    XT = (samples - xbar)/B**0.5  # shape BxD                                                                                                                                                
    GT = (vs - gbar)/B**0.5    # shape BxD                                                                                                                  
    Q = jnp.concatenate([reg**0.5*GT.T,  ((reg)/(1+reg))**0.5 * gbar.reshape(-1, 1)], axis=1) #Dx(B+1)
    R = jnp.concatenate([llambda0, reg**0.5*XT.T, (reg/(1+reg))**0.5*(mu0-xbar).reshape(-1, 1)], axis=1) #Dx(K+B+1)
    QTV = (Q.T@R)@R.T + (Q.T*psi0) #(B+1)xD
    Id_Q = jnp.identity(QTV.shape[0])
    M = 0.5*Id_Q + get_sqrt(0.25*Id_Q + QTV@Q).real
    #MM = jnp.linalg.pinv(M@M)
    #VTQM = (QTV).T@MM
    VTQM = jnp.linalg.solve((M@M).T, QTV).T

    mu = 1/(1+reg) * mu0 + reg/(1+reg) * (psi0*gbar + R@(R.T@gbar) - VTQM@(QTV@gbar) + xbar)    
    components = [R, VTQM, QTV]
    return mu, components



    
class PBAM:
    """
    Wrapper class for using GSM updates to fit a distribution
    """
    def __init__(self, D, lp, lp_g):
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
        


    def fit(self, key, regf, rank, mean=None, psi=None, llambda=None,
            batch_size=2, niter=5000, nprint=10, niter_em=10, jit_compile=True,
            tolerance=0, print_convergence=False, reg_factor=1., eta=1.,
            scalellambda = 1., tol_factor=1.,
            verbose=True, check_goodness=True, monitor=None, retries=10, jitter0=1e-6, jitterf=None):
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

        nevals = 1
        nprojects = []
        update_function = pbam_update
        project_function = partial(update_psi_llambda, jit_compile=jit_compile)
        if jit_compile:
            update_function = jit(update_function)
        if (nprint > niter) and verbose: nprint = niter
        if jitterf is None: jitterf = lambda x : jitter0

        # start loop
        if monitor is not None:
            monitor.means = []
            monitor.psis = []
            monitor.llambdas = []
            monitor.iparams = []
            monitor.nprojects = []
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
            jitter = jitterf(i)
            mean_prev, psi_prev, llambda_prev = None, None, None
            while True:         # Sometimes run crashes due to a bad sample. Avoid that by re-trying.
                try:
                    key, key_sample = random.split(key, 2)
                    np.random.seed(key_sample[0])

                    eps = np.random.normal(0, 1, size=(batch_size, self.D))
                    z = np.random.normal(0, 1, size=(batch_size, K))
                    if i == 0: samples = eps
                    else: samples = mean + psi**0.5 * eps + (llambda@z.T).T
                    vs = self.lp_g(samples)
                    nevals += batch_size
                    reg = regf(i) * reg_factor
                    mean_new, components = update_function(samples, vs, mean, psi, llambda, reg) # bam
                    # reset parameters
                    psi0 = psi.copy()
                    #R, VTQM, QTV = components
                    #llambda = np.random.normal(0, 1, size=(self.D, K)) *scalellambda
                    #psi = 1 + np.random.random(self.D)
                    #psi = psi + get_diag(R, R) - get_diag(VTQM, QTV.T) #- get_diag(llambda, llambda) #reset2
                    
                    psi_new, llambda_new, counter = project_function(psi, llambda, *components, \
                                                            niter_em=niter_em, tolerance=tolerance*tol_factor, eta=eta,
                                                                     verbose=print_convergence, psi0=psi0, jitter=jitter) # project
                    if i == 0: print('compiled')
                    monitor.nprojects.append(counter)
                    psi_new = jnp.maximum(psi_new, jitter) # jitter diagonal
                    # or update mean later?
                    #xbar = samples.mean(axis=0)
                    #gbar = vs.mean(axis=0)
                    #mean_new = 1/(1+reg) * mean + reg/(1+reg) * (psi_new*gbar + llambda_new@(llambda_new.T@gbar) + xbar)
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
                monitor_lr(monitor, i, [mean, psi, llambda], self.lp, key, nevals=nevals, force_save=True)
                monitor.means.append(mean)
                monitor.psis.append(psi)
                monitor.llambdas.append(llambda)
                monitor.iparams.append(i)
                return mean, psi, llambda
            
            elif is_good == 1:
                reg_factor = 1.
                tol_factor = 1.
                mean_prev, psi_prev, llambda_prev = mean*1., psi*1., llambda.copy()
                mean, psi, llambda = mean_new, psi_new, llambda_new
            else:
                reg_factor /= 2.
                if mean_prev is not None:
                    mean, psi, llambda = mean_prev, psi_prev, llambda_prev
                #tol_factor /= 10
                #if reg_factor < 2**-15:
                #    print("reg factor is very small")
                #    return mean, psi, llambda
                if verbose: print("Bad update for covariance matrix. Revert")

        if monitor is not None:
            monitor_lr(monitor, i, [mean, psi, llambda], self.lp, key, nevals=nevals, force_save=True)
            monitor.means.append(mean)
            monitor.psis.append(psi)
            monitor.llambdas.append(llambda)
            monitor.iparams.append(i)

        print('Total number of projections : ', np.sum(monitor.nprojects), np.sum(monitor.nprojects)/i)
        return mean, psi, llambda


    def _check_goodness(self, mean, psi, cov):
        is_good = 0
        j = 0 
        try:
            for m in [mean, psi, cov]:
                if (np.isnan(m)).any():
                    self.nan_update.append(j)
                    if len(self.nan_update) > 20:
                        is_good = -1
                    return is_good
            is_good = 1
            return is_good
        except:
            return is_good
        

        
class PBAM_fullcov:
    """
    Wrapper class for using GSM updates to fit a distribution
    """
    def __init__(self, D, lp, lp_g,  jit_compile=True):
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
        self.jit_compile = jit_compile
        if not jit_compile:
            print("Not using jit compilation. This may take longer than it needs to.")



    def fit(self, key, regf, rank, mean=None, cov=None,
            batch_size=2, niter=5000, nprint=10, n_project=128, tolerance=0., niter_em=10,
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

        if mean is None:
            mean = jnp.zeros(self.D)
        if cov is None:
            cov = jnp.identity(self.D)

        llambda, psi = None, None
        nevals = 1

        if batch_size < self.D:
            update_function = bam_lowrank_update
        else:
            update_function = bam_update
        if self.jit_compile:
            update_function = jit(update_function)
                
        if (nprint > niter) and verbose: nprint = niter
        for i in range(niter + 1):
            if (i%(niter//nprint) == 0) and verbose :
                print(f'Iteration {i} of {niter}')

            if monitor is not None:
                if (i%monitor.checkpoint) == 0:
                    monitor(i, [mean, cov], self.lp, key, nevals=nevals)
                    nevals = 0

            # Can generate samples from jax distribution (commented below), but using numpy is faster
            j = 0
            while True:         # Sometimes run crashes due to a bad sample. Avoid that by re-trying.
                try:
                    key, key_sample = random.split(key, 2)
                    np.random.seed(key_sample[0])
                    samples = np.random.multivariate_normal(mean=mean, cov=cov, size=batch_size)
                    # samples = MultivariateNormal(loc=mean, covariance_matrix=cov).sample(key, (batch_size,))
                    vs = self.lp_g(samples)
                    nevals += batch_size
                    reg = regf(i)
                    mean_new, cov_new = update_function(samples, vs, mean, cov, reg)
                    cov_new += np.eye(self.D) * jitter # jitter covariance matrix
                    cov_new = (cov_new + cov_new.T)/2.
                    break
                except Exception as e:
                    if j < retries :
                        j += 1
                        print(f"Failed with exception {e}")
                        print(f"Trying again {j} of {retries}")
                    else : raise e

            is_good = self._check_goodness(cov_new)
            if is_good:
                mean, cov = mean_new, cov_new
            else:
                if verbose: print("Bad update for covariance matrix. Revert")
                
            if tolerance!= 0: x = np.random.multivariate_normal(mean, cov, n_project)
            else: x = None
            if psi is None: 
                psi = jnp.diag(np.random.random(self.D))
                #psi = jnp.diag(jnp.diag(cov))
            if llambda is None: 
                llambda = np.random.normal(0, 1, size=(self.D, rank))
                #llambda = np.linalg.eigh(cov)[1][:, :rank]
            llambda, psi =  project_lr_gaussian(mean, cov, llambda, psi, data=x, num_of_itr=niter_em, tolerance=tolerance)
            psi += jitter
            #_, llambda, psi = fit_lr_gaussian(x, rank, verbose=False,
            #                                     mu=mean, llambda=llambda, psi=psi)
            cov = llambda@llambda.T + psi
        

        if monitor is not None:
            monitor(i, [mean, cov], self.lp, key, nevals=nevals)
        return mean, cov


    def _check_goodness(self, cov):
        '''
        Internal function to check if the new covariance matrix is a valid covariance matrix.
        Required due to floating point errors in updating the convariance matrix directly,
        insteead of it's Cholesky form.
        '''
        is_good = False
        try:
            if (np.isnan(np.linalg.cholesky(cov))).any():
                nan_update.append(j)
            else:
                is_good = True
            return is_good
        except:
            return is_good





# def bam_lowrank_update2(samples, vs, mu0, S0, reg):
#     """
#     Returns updated mean and covariance matrix with GSM updates.
#     For a batch, this is simply the mean of updates for individual samples.

#     Inputs:
#     samples: Array of samples of shape BxD where B is the batch dimension
#       vs : Array of score functions of shape BxD corresponding to samples
#       mu0 : Array of shape D, current estimate of the mean
#       S0 : Array of shape DxD, current estimate of the covariance matrix

#     Returns:
#       mu : Array of shape D, new estimate of the mean
#       S : Array of shape DxD, new estimate of the covariance matrix
#     """
    
#     assert len(samples.shape) == 2
#     assert len(vs.shape) == 2
#     B, D = samples.shape
#     xbar = jnp.mean(samples, axis=0)
#     gbar = jnp.mean(vs, axis=0)
#     XT = (samples - xbar)/B  # shape BxD
#     GT = (grads - gbar)/B    # shape BxD

#     U = np.stack([reg*GT,  (reg)/(1+reg) * gbar], axis=0)
#     #outer_map = jax.vmap(jnp.outer, in_axes=(0, 0))
#     #xdiff = samples - xbar
#     #C = jnp.mean(outer_map(xdiff, xdiff), axis=0)

#     gdiff = vs - gbar
#     G = jnp.mean(outer_map(gdiff, gdiff), axis=0)

#     U = reg * G + (reg)/(1+reg) * jnp.outer(gbar, gbar)
#     V = S0 + reg * C + (reg)/(1+reg) * jnp.outer(mu0 - xbar, mu0 - xbar)

#     # Form decomposition that is D x K
#     Q = compute_Q((U, B))
#     I = jnp.identity(B)
#     VT = V.T
#     A = VT.dot(Q)
#     BB = 0.5*I + jnp.real(get_sqrt(A.T.dot(Q) + 0.25*I))
#     BB = BB.dot(BB)
#     CC = jnp.linalg.solve(BB, A.T)
#     S = VT - A @ CC
#     mu = 1/(1+reg) * mu0 + reg/(1+reg) * (jnp.matmul(S, gbar) + xbar)

#     return mu, S

        
