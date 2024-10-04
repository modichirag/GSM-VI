'''Implement the EM algorithm for fitting low-rank + diagonal Gaussian
to data generated from a full rank Gaussian based on https://cseweb.ucsd.edu/~saul/papers/fa_ieee99.pdf
Convergence monitoring taken from
https://github.com/je-suis-tm/machine-learning/blob/master/factor%20analysis.ipynb
'''

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from numpyro.distributions import Normal

@jit
def get_latent(data,miu,lambda_,psi):
    
    psi_inv = jnp.diag(jnp.diagonal(psi)**-1)
    weighted_psi = lambda_.T@psi_inv
    cov_z = jnp.linalg.inv(jnp.identity(lambda_.shape[1])+weighted_psi@lambda_)
    z= (data - miu)@(weighted_psi).T@cov_z
    
    return z


@jit
def get_lower_bound(data, mu, llambda, psi):
    
    z = get_latent(data, mu, llambda, psi)    
    loc = (llambda@z.T).T
    scale = jnp.diag(psi)**0.5
    logpdf = Normal(loc, scale).log_prob(data)
    lower_bound = logpdf.sum()
    
    return lower_bound



@jit
def estep(x, mu, llambda, psi):
    print('jit estep')
    D = mu.size
    r = llambda.shape[1]
    psi_inv = jnp.diag(jnp.diagonal(psi)**-1)
    alpha = llambda.T@psi_inv    
    beta = jnp.linalg.inv(np.identity(r) + alpha@llambda)
    e_zmu_x = (beta@alpha)@(x-mu).T
    e_zsigma_x = beta
    return e_zmu_x.T, e_zsigma_x

@jit
def mstep(x, mu, llambda, psi, e_zmu_x, e_zsigma_x):
    print('jit mstep')
    N, D, r = x.shape[0], x.shape[1], llambda.shape[1]    
    delx = x - x.mean(axis=0)
    delz = e_zmu_x  - e_zmu_x.mean(axis=0)

    zcov = jnp.zeros_like(e_zsigma_x)
    xcov = jnp.zeros((D, r))
    # for z in delz:
    #     zcov += jnp.outer(z, z)/N
    # for a, b in zip(delx, delz):
    #     xcov += jnp.outer(a, b)/N
    ## vamp will blow up memory here, need running average
    zcov = jnp.mean(jax.vmap(jnp.outer, in_axes=[0, 0])(delz, delz), axis=0)
    xcov = jnp.mean(jax.vmap(jnp.outer, in_axes=[0, 0])(delx, delz), axis=0)
    
    llambda_update = xcov @ jnp.linalg.inv(e_zsigma_x + zcov)
    mu_update = (x - (llambda_update@e_zmu_x.T).T).mean(axis=0)
    psi_update = jnp.diag(((delx - (llambda_update@delz.T).T)**2).mean(axis=0) + jnp.diagonal(llambda_update@e_zsigma_x@llambda_update.T))
    
    return mu_update, llambda_update, psi_update



#using mle for training
def fit_lr_gaussian(data, num_of_latents,
                    mu=None, llambda=None, psi=None,
                    tolerance=0.001, niters=100,
                    verbose=False):
        
    #get dimensions
    D = data.shape[1]
    
    #initialize
    lower_bound_old=None
    lower_bound=None
    counter=0
    
    #miu is the unconditional mean
    if mu is None:
        mu = data.mean(axis=0)    
    
    #lambda_ is the coefficient of the latent variables
    #use principal components to initialize
    #this approach improves the performance (Barber 2012)
    if llambda is None:
        _, eigvecs = np.linalg.eig((data-mu).T@(data-mu)/data.shape[0])
        llambda = eigvecs[:, :num_of_latents]
    
    #psi is the covariance matrix of noise, a diagonal matrix
    #use the diagonal of covariance matrix to initialize
    if psi is None:
        psi = np.diag(np.diag((data - mu).T@(data - mu)/data.shape[0]))  

    #cap the maximum number of iterations
    while counter < niters:
        
        e_zmu_x, e_zsigma_x = estep(data, mu, llambda, psi)
        mu, llambda, psi = mstep(data, mu, llambda, psi, e_zmu_x, e_zsigma_x)
            
        #use lower bound to determine if convergeda
        lower_bound_old=lower_bound
        lower_bound=get_lower_bound(data, mu, llambda, psi)
            
        if lower_bound_old and np.abs(lower_bound/lower_bound_old-1)<tolerance:
            if verbose:
                print(f'{counter} iterations to reach convergence\n')
            return mu, llambda, psi
            
        counter+=1
    
    if verbose:
        print(f'{counter} iterations to reach convergence\n')
        
    return mu, llambda, psi



# @jit
# def em_exact(llambda, psi, mu, cov):
#     D, r = llambda.shape#[0], llambda.shape[1]
#     psi_inv = jnp.diag(jnp.diagonal(psi)**-1)
#     alpha = llambda.T@psi_inv    
#     beta = jnp.linalg.pinv(np.identity(r) + alpha@llambda)

#     gamma = cov@(alpha.T@beta.T)
#     llambda_update = gamma @ jnp.linalg.pinv(beta + beta@alpha@gamma) 
    
#     A = jnp.eye(D) - llambda_update @beta @ llambda.T @psi_inv
#     M = A@cov@A.T + llambda_update @beta @llambda_update.T
#     psi_update = jnp.diag(jnp.diagonal(M))
#     return llambda_update, psi_update


# #using mle for training
# def project_lr_gaussian(mu, cov, llambda, psi, num_of_itr=10, 
#                         tolerance=0.001, diagnosis=False, 
#                         data=None):
        
#     #cap the maximum number of iterations
#     #initialize
#     lower_bound_old=None
#     lower_bound=None
#     counter=0
#     while counter < num_of_itr:
        
#         llambda, psi = em_exact(llambda, psi, mu, cov)
            
#         #use lower bound to determine if converged if data is given
#         if data is not None:
#             lower_bound_old=lower_bound
#             lower_bound=get_lower_bound(data, mu, llambda, psi)            
#             if lower_bound_old and np.abs(lower_bound/lower_bound_old-1)<tolerance:
#                 if diagnosis:
#                     print(f'{counter} iterations to reach convergence\n')
#                 return llambda, psi
            
#         counter+=1
    
#     if diagnosis:
#         print(f'{counter} iterations to reach convergence\n')
        
#     return llambda, psi



@jit
def em_exact(llambda, psi, cov):
    D, K = llambda.shape
    psi_inv = jnp.diag(jnp.diagonal(psi)**-1)
    alpha = llambda.T@psi_inv    
    beta = jnp.linalg.pinv(np.identity(K) + alpha@llambda)

    gamma = cov@(alpha.T@beta.T)
    llambda_update = gamma @ jnp.linalg.pinv(beta + beta@alpha@gamma) 
    A = jnp.eye(D) - llambda_update @beta @ llambda.T @psi_inv
    M = A@cov@A.T + llambda_update @beta @llambda_update.T
    psi_update = jnp.diag(jnp.diagonal(M))
    
    return llambda_update, psi_update



@jit
def lower_bound_kl(cov, psi, llambda):
    A = psi + llambda@llambda.T #+ 1e-6*np.eye(psi.size)
    logabsdet = jnp.linalg.slogdet(A)[1]
    kl = jnp.trace(cov@jnp.linalg.pinv(A)) + logabsdet
    return kl



def fit_lr_gaussian_law(cov, num_of_latents, eta=1.0,
                tolerance=0.001,num_of_itr=100, diagnosis=False, 
                        llambda=None, psi=None, nprint=100):
        

    D = cov.shape[0]
    K = num_of_latents
    if llambda is None:
        llambda = np.random.normal(0, 1, size=(D, K))
    if psi is None:
        psi = np.diag(np.random.random(D))


    #initialize
    lower_bound_old=None
    lower_bound=None
    counter=0
    losses = []
    psis, llambdas = [], []
    llambdas.append(llambda)
    psis.append(np.diag(psi))
    lower_bound=lower_bound_kl(cov, psi, llambda)
    losses.append(lower_bound)
        
    while counter < num_of_itr:

        if (counter%(num_of_itr//nprint) == 0) : print(counter, lower_bound)
        llambda_update, psi_update = em_exact(llambda, psi,  cov)
        psi = (1 - eta)*psi + eta*psi_update
        llambda = (1 - eta)*llambda  + eta*llambda_update
        
        #use lower bound to determine if convergeda
        lower_bound_old=lower_bound
        lower_bound=lower_bound_kl(cov, psi, llambda)
            
        if lower_bound_old and np.abs(lower_bound/lower_bound_old-1)<tolerance:
            if diagnosis:
                print(f'{counter} iterations to reach convergence\n')
            return psi, llambda, [psis, llambdas, losses]
            
        llambdas.append(llambda)
        psis.append(np.diag(psi))
        losses.append(lower_bound)
        counter+=1
    
    if diagnosis:
        print(f'{counter} iterations to reach convergence\n')
        
    return psi, llambda, [psis, llambdas, losses]


@jit
def get_diag(u, v):
        return jax.vmap(jnp.dot, in_axes=[0, 0])(u, v)
    


@jit
def em_diana(llambda, psi, cov):
    D, K = llambda.shape
    psi_inv = jnp.diag(jnp.diagonal(psi)**-1)
    Id_K = jnp.identity(K)
    Id_D = jnp.identity(D)
    
    alpha = llambda.T@psi_inv 
    beta = alpha @ (Id_D - llambda@jnp.linalg.pinv(Id_K + alpha@llambda) @alpha)    
    SBT = (cov@beta.T)
    llambda_update = SBT@jnp.linalg.pinv(beta@SBT + Id_K - beta@llambda)
    psi_update = jnp.diag(jnp.diagonal((cov - llambda_update@SBT.T)))
    
    return llambda_update, psi_update
                          
    
    
def fit_lr_gaussian_diana(cov, num_of_latents, eta=1.0,
                tolerance=0.001,num_of_itr=100, diagnosis=False, 
                          llambda=None, psi=None, nprint=100):

    D = cov.shape[0]
    K = num_of_latents
    if llambda is None:
        llambda = np.random.normal(0, 1, size=(D, K))
    if psi is None:
        psi = np.diag(np.random.random(D))


    #cap the maximum number of iterations
    #initialize
    lower_bound_old=None
    lower_bound=None
    counter=0
    losses = []
    psis, llambdas = [], []
    
    llambdas.append(llambda)
    psis.append(np.diag(psi))
    lower_bound=lower_bound_kl(cov, psi, llambda)
    losses.append(lower_bound)
    
    while counter < num_of_itr:

        if (counter%(num_of_itr//nprint) == 0) : print(counter, lower_bound)
        llambda_update, psi_update = em_diana(llambda, psi, cov)
        psi = (1 - eta)*psi + eta*psi_update
        llambda = (1 - eta)*llambda  + eta*llambda_update

        #use lower bound to determine if convergeda
        lower_bound_old=lower_bound
        lower_bound=lower_bound_kl(cov, psi, llambda)
            
        if lower_bound_old and np.abs(lower_bound/lower_bound_old-1)<tolerance:
            if diagnosis:
                print(f'{counter} iterations to reach convergence\n')
            return psi, llambda, [psis, llambdas, losses]
            
        llambdas.append(llambda*1.)
        psis.append(np.diag(psi))
        losses.append(lower_bound)
        counter+=1
    
    if diagnosis:
        print(f'{counter} iterations to reach convergence\n')
        
    return psi, llambda, [psis, llambdas, losses]
