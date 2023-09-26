
###Functions for Gaussian score matching

import numpy as np
import tensorflow as tf
import divergences as divs
import sys, os
from modevi import hill_climb

@tf.function
def get_update(sample, v, mu0, S0):
    '''returns GSM update to mean and covariance matrix for a single sample
    '''
    vSv = tf.linalg.einsum('i,i->', v, tf.linalg.einsum('ij,j->i', S0, v))
    mu_v = tf.linalg.einsum('i,i->', (mu0 - sample), v)
    rho = 0.5 * tf.sqrt(1 + 4*(vSv + mu_v**2)) - 0.5
    eps0 = tf.linalg.einsum('ij,j->i', S0, v) - mu0 + sample

    #mu update
    mu_vT = tf.linalg.einsum('i,j->ij', (mu0 - sample), v)
    den = 1 + rho + mu_v
    I = tf.eye(sample.shape[0], dtype=sample.dtype)
    mu_update = 1/(1 + rho) * tf.linalg.einsum('ij, j->i', ( I - mu_vT / den), eps0)
    mu = mu0 + mu_update

    #S update
    Supdate_0 =  tf.linalg.einsum('i,j-> ij', (mu0-sample), (mu0-sample))
    Supdate_1 =  tf.linalg.einsum('i,j-> ij', (mu-sample), (mu-sample))
    S_update = (Supdate_0 - Supdate_1)
    # S_update = (Supdate_0 - Supdate_1 + tf.linalg.eye(S0.shape[0], dtype=dtype)*0.)
    return mu_update, S_update



#@tf.function #otherwise we get tensor has no .numpy() error
def gaussian_update_batch(samples, model, qmodel, nexception=0):
    '''returns GSM update to mean and covariance matrix for a batch of samples. 
    This calls single sample update for each sample.
    '''
    #print("creating graph gaussian_update_batch")
    S0 = tf.identity(qmodel.cov)
    mu0 = tf.identity(qmodel.loc*1.)
    
    #intermediate variables
    mu_updates, S_updates = [], []
    for i in range(len(samples)):
        sample = samples[i]
        v = model.grad_log_likelihood(sample)
        mu_update, S_update = get_update(sample, v, mu0, S0)
        mu_updates.append(mu_update)
        S_updates.append(S_update)
        
    mu_update = tf.reduce_mean(tf.stack(mu_updates, axis=0), axis=0)
    S_update = tf.reduce_mean(tf.stack(S_updates, axis=0), axis=0)
    mu = mu0 + mu_update
    S = S0 + S_update
    
    #Exception handling 
    try:
        if np.linalg.eigvals(S).min() > 0: # everything seems fine, update distribution
            qmodel.loc.assign(mu)
            qmodel.cov.assign(S)
            nexception = 0
            return nexception
        else:
            print(f"Exception {nexception} : NEGATIVE EIGENVALUES IN COVARIANCE.\nTry again")
            nexception +=1
            return nexception
    except Exception as e:
            print(f"Exception {nexception} : {e}")
            nexception +=1
            return nexception
        
        
def train(qdist, model, niter=1000, batch_size=4, nprint=10, dtype=tf.float32, callback=None, verbose=True, samples=None,
          nqsamples=1000, ntries=10, savepath=None):
    '''Main function for training variational distribution (qdist) to match the target distribution (model) with score matching using GSM.
    Returns fit variational distribution and other diagnostics
    '''

    qdivs = []
    counts = []
    fdivs = []
    nexception = 0 

        
    for epoch in range(niter + 1):

        if (epoch %(niter//nprint) == 0) & verbose: 
            print(f"Iteration {epoch}")
            
        #measure divergences
        qdiv = divs.qdivergence(qdist, model, nsamples=nqsamples)
        fdiv = divs.fkl_divergence(qdist, model, samples=samples)
        qdivs.append(qdiv)
        fdivs.append(fdiv)
            
        #grad and update
        run = 1        
        for itry in range(ntries+1):
            try:
                x = list(qdist.sample(batch_size))        
                nexception = gaussian_update_batch(x, model, qdist, nexception)
                if nexception == 0 :
                    run = 0
            except Exception as e:
                print(f"Iteration {epoch}, try {itry}, exception in iteration for : \n" , e)
            if run == 0 :
                break
        if (itry == ntries) :
            print("Max iterations reached")
            raise Exception
        
        counts.append(model.grad_count)
        
        if (epoch %(niter//nprint) == 0) & verbose: 
            print("Loss at epoch %d is"%epoch, qdiv[0])
            sys.stdout.flush()
            if callback is not None: 
                callback(qdist, [np.array(qdivs)[:, 0]], epoch)
            if savepath is not None:
                np.save(savepath + 'qdivs', np.array(qdivs))
                np.save(savepath + 'counts', np.array(counts))
                np.save(savepath + 'fdivs', np.array(fdivs))
                qsamples = qdist.sample(1000)
                qsamples = model.constrain_samples(qsamples).numpy()
                np.save(savepath + 'samples', qsamples)


    return qdist, np.array(qdivs), np.array(counts), np.array(fdivs)
    
