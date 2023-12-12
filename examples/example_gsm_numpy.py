## A basic example for fitting a target Multivariate Gaussian distribution with GSM updates

import numpy as np

from gsmvi.gsm_numpy import GSM

#####
def setup_model(D):
   
    # setup a Gaussian target distribution
    mean = np.random.random(D)
    L = np.random.normal(size = D**2).reshape(D, D)
    cov = np.matmul(L, L.T) + np.eye(D)*1e-3
    icov = np.linalg.inv(cov)

    # functions for log_prob and score. These are to be supplied by the user
    def lp(x):
        assert len(x.shape) == 2
        lp = 0
        for i in range(x.shape[0]):
            lp +=  -0.5 * np.dot(np.dot(mean - x[i], icov), mean - x[i])
        return lp

    def lp_g(x):
        assert len(x.shape) == 2
        lp_g = []
        for i in range(x.shape[0]):
            lp_g.append( -1. * np.dot(icov, x[i] - mean))
        return np.array(lp_g)
        
    return mean, cov, lp, lp_g


if __name__=="__main__":
    
    ###
    # setup a toy Gaussia model and extracet score needed for GSM
    D =50
    mean, cov, lp, lp_g = setup_model(D=D)

    ###
    # Fit with GSM
    niter = 500
    key = 99
    gsm = GSM(D=D, lp=lp, lp_g=lp_g)
    mean_fit, cov_fit = gsm.fit(key, niter=niter)

    print("\nTrue mean : ", mean)
    print("Fit mean  : ", mean_fit)
