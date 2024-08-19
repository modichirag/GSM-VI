import jax
import jax.numpy as jnp
from jax import jit, random
from jax.scipy.linalg import sqrtm as sqrtm_jsp
from scipy.linalg import sqrtm as sqrtm_sp
#from numpyro.distributions import MultivariateNormal  ##Needed if sampling from numpyro dist below
import numpy as np
import scipy.sparse as spys
from jax.lib import xla_bridge
from em_lr_projection import fit_lr_gaussian, project_lr_gaussian

from bam import bam_update, bam_lowrank_update, get_sqrt
from functools import partial


@jit
def det_cov_lr(psi, llambda):
    m = (llambda.T*(1/psi))@llambda
    m = np.identity(m.shape[0]) + m
    return jnp.linalg.det(m)*jnp.prod(psi)

@jit
def logp_lr(y, mean, psi, llambda):

    D, K = llambda.shape
    x = y - mean
    
    first_term = jnp.dot(x, x/psi)
    ltpsinv = llambda.T*(1/psi)
    m = jnp.identity(K) + ltpsinv@llambda
    minv = jnp.linalg.inv(m)
    res = ltpsinv@x
    second_term = res.T@minv@res
    
    logexp = -0.5 * (first_term - second_term)
    logdet = - 0.5 * jnp.log(jnp.linalg.det(m)*jnp.prod(psi))
    logp = logexp + logdet - 0.5*D*jnp.log(2*jnp.pi)
    return logp


@jit
def get_diag(U, V):
    """Return diagonal of U@V.T"""
    return jax.vmap(jnp.dot, in_axes=[0, 0])(U, V)


#@jit
def _update_psi_llambda(psi, llambda, R, VTQM, QTV, psi0, first_term_1):
    print('jit psi/llambda update')
    # first update llambda
    D, K = llambda.shape
    Id_K = jnp.identity(K)
    psi_inv = psi**-1
    C = jnp.linalg.pinv(Id_K + (llambda.T *psi_inv)@llambda) #KxK  #replace by Solve?
    J = C@(llambda.T*psi_inv) #KxD
    VJT = (J*psi0).T + R@(R.T@J.T) #DxK
    SJT = VJT - VTQM@(QTV@J.T)   #SigmaJT , DxK
    llambda_update = SJT @ jnp.linalg.pinv(C + J@SJT)  #replace by Solve?

    # now update psi.
    first_term_2 = -2* get_diag(SJT, llambda_update)
    Mint = J@SJT
    def contractM(a):
        return a@Mint@a
    first_term_3 = jax.vmap(contractM, in_axes=[0])(llambda_update) 
    first_term = first_term_1 + first_term_2 + first_term_3
    second_term = get_diag(llambda_update @C , llambda_update)
    psi_update = first_term + second_term

    return psi_update, llambda_update


def update_psi_llambda(psi, llambda, R, VTQM, QTV, niter_em, jit_compile=True):

    first_term_1 = psi + get_diag(R, R) - get_diag(VTQM, QTV.T) #diag of cov for psi
    psi0 = psi.copy()     
    if jit_compile :  _update = jit(_update_psi_llambda)
    else:   _update = _update_psi_llambda
    counter = 0
    
    for counter in range(niter_em):
        psi, llambda = _update(psi, llambda, R, VTQM, QTV, psi0, first_term_1)
        
    return psi, llambda



#@jit
def pbam_update(samples, vs, mu0, psi0, llambda0, reg,  niter_em=2):
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
    VTQ = (Q.T*psi0).T + R@(R.T@Q) #Dx(B+1)
    QTV = (Q.T@R)@R.T + (Q.T*psi0) #(B+1)xD
    Id_Q = jnp.identity(QTV.shape[0])
    M = 0.5*Id_Q + get_sqrt(0.25*Id_Q + QTV@Q).real
    MM = jnp.linalg.pinv(M@M)
    VTQM = VTQ@MM

    mu = 1/(1+reg) * mu0 + reg/(1+reg) * (psi0*gbar + R@(R.T@gbar) - VTQM@(QTV@gbar) + xbar)    
    components = [R, VTQM, QTV]
    return mu, components



class PBAM:
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
        


    def fit(self, key, regf, rank, mean=None, psi=None, llambda=None,
            batch_size=2, niter=5000, nprint=10, niter_em=10, jit_compile=True,
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
            llambda = np.random.normal(0, 1, size=(self.D, K))
        if psi is None:
            psi = np.random.random(self.D)

        nevals = 1
        update_function = pbam_update
        project_function = partial(update_psi_llambda, jit_compile=jit_compile)
        if jit_compile:
            update_function = jit(update_function)
        if (nprint > niter) and verbose: nprint = niter
        for i in range(niter + 1):
            if (i%(niter//nprint) == 0) and verbose :
                print(f'Iteration {i} of {niter}')

            if monitor is not None:
                pass
                # if (i%monitor.checkpoint) == 0:
                #     monitor(i, [mean, cov], self.lp, key, nevals=nevals)
                #     nevals = 0

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
                    reg = regf(i)
                    mean, components =  update_function(samples, vs, mean, psi, llambda, reg, niter_em=niter_em) # bam
                    psi, llambda = project_function(psi, llambda, *components, niter_em) # project
                    psi +=  jitter # jitter diagonal
                    break
                except Exception as e:
                    if j < retries :
                        j += 1
                        print(f"Failed with exception {e}")
                        print(f"Trying again {j} of {retries}")
                    else : raise e

            # is_good = self._check_goodness(cov_new)
            # if is_good:
            #     mean, cov = mean_new, cov_new
            # else:
            #     if verbose: print("Bad update for covariance matrix. Revert")
                
            # x = np.random.multivariate_normal(mean, cov, n_project)
            # if psi is None: 
            #     psi = jnp.diag(jnp.diag(cov))
            # if llambda is None: 
            #     llambda = np.linalg.eigh(cov)[1][:, :rank]
            # llambda, psi =  project_lr_gaussian(mean, cov, llambda, psi, data=x)
            # #_, llambda, psi = fit_lr_gaussian(x, rank, verbose=False,
            # #                                     mu=mean, llambda=llambda, psi=psi)
            # cov = llambda@llambda.T + psi
        

        if monitor is not None:
            pass
            #monitor(i, [mean, cov], self.lp, key, nevals=nevals)
            
        return mean, psi, llambda

    

class PBAM_fullcov:
    """
    Wrapper class for using GSM updates to fit a distribution
    """
    def __init__(self, D, lp, lp_g, use_lowrank=False, jit_compile=True):
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
        self.use_lowrank = use_lowrank
        if use_lowrank:
            print("Using lowrank update")
        self.jit_compile = jit_compile
        if not jit_compile:
            print("Not using jit compilation. This may take longer than it needs to.")



    def fit(self, key, regf, rank, mean=None, cov=None,
            batch_size=2, niter=5000, nprint=10, n_project=128,
            verbose=True, check_goodness=True, monitor=None, retries=10, jitter=1e-6, early_stop=True):
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

        if self.use_lowrank:
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
                
            if early_stop: x = np.random.multivariate_normal(mean, cov, n_project)
            else: x = None
            if psi is None: 
                psi = jnp.diag(np.random.random(self.D))
                #psi = jnp.diag(jnp.diag(cov))
            if llambda is None: 
                llambda = np.random.normal(0, 1, size=(self.D, rank))
                #llambda = np.linalg.eigh(cov)[1][:, :rank]
            llambda, psi =  project_lr_gaussian(mean, cov, llambda, psi, data=x)
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

        
