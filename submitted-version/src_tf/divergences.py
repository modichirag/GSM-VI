###Functions for forward and backward KL divergence to monitor divergences in training
import numpy as np
import tensorflow as tf


#@tf.function
def qdivergence(qdist, model, nsamples=32):
    '''Estimates backward KL divergence (samples generated from variational distribution)
    '''
    try:
        samples = qdist.sample(nsamples)
        logp1 = qdist.log_prob(samples)
        logp2 = model.log_prob(samples)
        div = tf.reduce_mean(logp1 - logp2)
        return div, tf.reduce_mean(logp1), tf.reduce_mean(logp2)
    
    except Exception as e:
        print("Exception in estimating qdivergence")
        print(e)
        return np.NaN, np.NaN, np.NaN
        


#@tf.function
def fkl_divergence(qdist, model=None, samples=None, nsamples=32):
    '''Estimates forward KL divergence (samples generated from target distribution)
    '''

    try:
        if samples is None:
            samples = model.sample(nsamples)
        logp1 = qdist.log_prob(samples)
        if model is not None:
            logp2 = model.log_prob(samples)
        else:
            logp2 = 0.
        div = tf.reduce_mean(logp1 - logp2)
        return div, tf.reduce_mean(logp1), tf.reduce_mean(logp2)

    except Exception as e:
        print("Exception in estimating fdivergence")
        print(e)
        return np.NaN, np.NaN, np.NaN
        
