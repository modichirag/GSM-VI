## Pure numpy implementation of GSM updates. 

import numpy as np

def _gsm_update_single(sample, v, mu0, S0):
    '''returns GSM update to mean and covariance matrix for a single sample
    '''
    S0v = np.matmul(S0, v)
    vSv = np.matmul(v, S0v)
    mu_v = np.matmul((mu0 - sample), v)
    rho = 0.5 * np.sqrt(1 + 4*(vSv + mu_v**2)) - 0.5
    eps0 = S0v - mu0 + sample

    #mu update
    mu_vT = np.outer((mu0 - sample), v)
    den = 1 + rho + mu_v
    I = np.eye(sample.shape[0])
    mu_update = 1/(1 + rho) * np.matmul(( I - mu_vT / den), eps0)
    mu = mu0 + mu_update

    #S update
    Supdate_0 =  np.outer((mu0-sample), (mu0-sample))
    Supdate_1 =  np.outer((mu-sample), (mu-sample))
    S_update = (Supdate_0 - Supdate_1)
    return mu_update, S_update


def gsm_update(samples, vs, mu0, S0):
    """
    Returns updated mean and covariance matrix with GSM updates.
    For a batch, this is simply the mean of updates for individual samples.

    Inputs:
      samples: Array of samples of shape BxD where B is the batch dimension
      vs : Array of score functions of shape BxD corresponding to samples
      mu0 : Array of shape D, current estimate of the mean
      S0 : Array of shape DxD, current estimate of the covariance matrix

    Returns:
      mu : Array of shape D, new estimate of the mean
      S : Array of shape DxD, new estimate of the covariance matrix
    """
        
    assert len(samples.shape) == 2
    assert len(vs.shape) == 2

    B, D = samples.shape
    mu_update, S_update = np.zeros((B, D)), np.zeros((B, D, D))
    for i in range(B):
        mu_update[i], S_update[i] = _gsm_update_single(samples[i], vs[i], mu0, S0)
    mu_update = np.mean(mu_update, axis=0)
    S_update = np.mean(S_update, axis=0)
    mu = mu0 + mu_update
    S = S0 + S_update

    return mu, S


