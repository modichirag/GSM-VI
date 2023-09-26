###Functions for Black box variational inference with reparametrization gradients

import numpy as np
import tensorflow as tf 
import divergences as divs
import sys, os



@tf.function
def bbvi_elbo(qdist, model, batch=tf.constant(32)):   
    '''For a given variational distribution (qdist) and target distribution (model)
    return elbo, negative elbo and gradient with respect to variational parameters
    '''
    with tf.GradientTape(persistent=True) as tape:
        print("creating graph bbvi_elbo/full")
        tape.watch(qdist.trainable_variables)
        sample = qdist.sample(batch) 
        try:
            logl = model.log_likelihood_and_grad(sample)
        except:
            logl = model.log_prob(sample)
        logq = qdist.log_prob(sample)
        elbo = logl - logq
        negelbo = -1. * tf.reduce_mean(elbo, axis=0)
        loss = negelbo
        
    gradients = tape.gradient(negelbo, qdist.trainable_variables)
    return tf.reduce_mean(elbo), loss, gradients



def train(qdist, model, lr=1e-3, batch=32, niter=1001, nprint=None, verbose=True, callback=None, samples=None, nqsamples=1000, savepath=None):
    '''Main function for training variational distribution (qdist) to match the target distribution (model)
    by minimizing elbo. Returns fit variational distribution and other diagnostics
    '''
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    elbos, losses, qdivs, counts, fdivs = [], [], [], [], []

    for epoch in range(niter+1):

        #compare
        qdiv = divs.qdivergence(qdist, model, nsamples=nqsamples)
        qdivs.append(qdiv)
        if samples is not None:
            fdiv = divs.fkl_divergence(qdist, model, samples=samples)
            fdivs.append(fdiv)
        else:
            fdivs = None

        #grad and update
        elbo, loss, grads = bbvi_elbo(qdist, model, batch=tf.constant(batch))
        elbo = elbo.numpy()
        if np.isnan(elbo):
            print("ELBO is NaNs :: ", epoch, elbo)
            break

        opt.apply_gradients(zip(grads, qdist.trainable_variables))
        elbos.append(elbo)
        losses.append(loss)
        counts.append(model.grad_count) 
        if (epoch %nprint == 0) & verbose: 
            print("Elbo at epoch %d is"%epoch, elbo)
            sys.stdout.flush()
            if callback is not None: callback(qdist, [np.array(elbos)], epoch)
            if savepath is not None:
                np.save(savepath + 'elbo', np.array(elbos))
                np.save(savepath + 'qdivs', np.array(qdivs))
                np.save(savepath + 'losses', np.array(losses))
                np.save(savepath + 'counts', np.array(counts))
                np.save(savepath + 'fdivs', np.array(fdivs))
                qsamples = qdist.sample(1000)
                qsamples = model.constrain_samples(qsamples).numpy()
                np.save(savepath + 'samples', qsamples)


    print("return")
    return qdist, np.array(losses), np.array(elbos), np.array(qdivs), np.array(counts), np.array(fdivs)
    

def train_frg(qdist, model, lr=1e-3, mode='full', batch=32, niter=1001, nprint=None, verbose=True, callback=None, samples=None, nqsamples=1000, savepath=None, ntries=10):
    '''Main function for training variational distribution (qdist) to match the target distribution (model)
    by minimizing elbo. Returns fit variational distribution and other diagnostics
    Different from train function as it has better handling of NaNs and infs. 
    '''
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    elbos, losses, qdivs, counts, fdivs = [], [], [], [], []

    for epoch in range(niter+1):

        #compare
        #measure divergences
        qdiv = divs.qdivergence(qdist, model, nsamples=nqsamples)
        fdiv = divs.fkl_divergence(qdist, model, samples=samples)
        qdivs.append(qdiv)
        fdivs.append(fdiv)
            
        #grad and update
        loc, scale = qdist.loc*1., qdist.scale*1.
        run = 1
        for itry in range(ntries):
            try:
                elbo, loss, grads = bbvi_elbo(qdist, model, batch=tf.constant(batch))
                if ~np.isnan(elbo) :
                    run = 0
            except Exception as e:
                print(f"Iteration {epoch}, try {itry}, exception in iteration for : \n" , e)
            if run == 0 :
                break
        if itry > ntries:
            print("Max iterations reached")
            raise Exception
        
        opt.apply_gradients(zip(grads, qdist.trainable_variables))
        try:
            sample = qdist.sample()
        except Exception as e:
            print('Exception in sampling after update, revert to previous point')
            qdist.loc.assign(loc)
            qdist.scale.assign(scale)
            print()

        #Logging
        elbos.append(elbo.numpy())
        losses.append(loss)
        counts.append(model.grad_count) 
        if (epoch %nprint == 0) & verbose: 
            print("Elbo at epoch %d is"%epoch, elbo)
            sys.stdout.flush()
            if callback is not None: callback(qdist, [np.array(elbos)], epoch)
            if savepath is not None:
                np.save(savepath + 'elbo', np.array(elbos))
                np.save(savepath + 'qdivs', np.array(qdivs))
                np.save(savepath + 'losses', np.array(losses))
                np.save(savepath + 'counts', np.array(counts))
                np.save(savepath + 'fdivs', np.array(fdivs))
                qsamples = qdist.sample(1000)
                qsamples = model.constrain_samples(qsamples).numpy()
                np.save(savepath + 'samples', qsamples)


    print("return")
    return qdist, np.array(losses), np.array(elbos), np.array(qdivs), np.array(counts), np.array(fdivs)
    
