import numpy as np
import tensorflow as tf 

@tf.function
def hill_climb(sample, log_likelihood, log_prior):
    '''Return loglikelihood and gradient for maximizing log-likelihood
    '''
    print('Hill climb')
    with tf.GradientTape(persistent=True) as tape_in:
        tape_in.watch(sample)
        logl = log_likelihood(sample)
        logpr = log_prior(sample)
        logp = logl + logpr
        f = -logp
        f = tf.reduce_mean(f)
    grad_sample = tape_in.gradient(f, sample)
    return f, grad_sample

    

