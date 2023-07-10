import numpy as np
from scipy.optimize import minimize


def lbfgs_init(x0, lp, lp_g=None, maxiter=1000, maxfun=1000):

    f = lambda x: -lp(x)
    if lp_g is not None:
        f_g = lambda x: -lp_g(x)
    else:
        f_g = None
    res = minimize(f, x0, method='L-BFGS-B', jac=f_g, \
                   options={"maxiter":maxiter, "maxfun":maxfun})

    mu = res.x
    cov = res.hess_inv.todense()
    return mu, cov, res



