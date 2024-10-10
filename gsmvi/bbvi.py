import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad, random
from numpyro.distributions import MultivariateNormal, Normal
import optax
from gsmvi.low_rank_gaussian import logp_lr, monitor_lr

@jit
def logp_lr_unstable(y, mean, psi, llambda):

    D, K = llambda.shape
    x = y - mean
    
    first_term = jnp.dot(x, x/psi)
    ltpsinv = llambda.T*(1/psi)
    m = jnp.identity(K) + ltpsinv@llambda
    #minv = jnp.linalg.pinv(m)
    res = ltpsinv@x
    #second_term = res.T@minv@res
    second_term = res.T@jnp.linalg.solve(m, res)
    
    logexp = -0.5 * (first_term - second_term)
    logdet = -0.5 * (jnp.linalg.slogdet(m)[1] + jnp.sum(jnp.log(psi))) #jnp.log(jnp.linalg.det(m)*jnp.prod(psi))
    #logdet = jnp.log(jnp.linalg.det(m)*jnp.prod(psi))
    logp = logexp + logdet - 0.5*D*jnp.log(2*jnp.pi)
    #logp = logexp - 0.5*D*jnp.log(2*jnp.pi)
    return logp


logp_lr_vmap = jax.vmap(logp_lr_unstable, in_axes=[0, None, None, None])



class ADVI_Factorized():
    def __init__(self, D, lp, lp_g=None, jit_compile=True):
        """
        Inputs:
          D: (int) Dimensionality (number) of parameters.
          lp : Function to evaluate target log-probability distribution
               whose gradient can be evaluated with jax.grad(lp)
        """
        self.D = D
        self.lp = lp
        self.lp_g = lp_g
        self.jit_compile = jit_compile
        if not jit_compile:
            print("Not using jit compilation. This may take longer than it needs to.")


    def loss_function(self, params, key, batch_size):
        """
        Internal function to evaluate negative-ELBO which is the loss function
        """
        loc, logscale = params
        scale = jnp.exp(logscale)
        q = Normal(loc=loc, scale=scale)
        #
        samples = q.sample(key, (batch_size,))
        logl = jnp.sum(self.lp(samples))
        logq = jnp.sum(q.log_prob(samples))
        elbo = logl - logq
        negelbo = -1. * elbo
        return negelbo

    
    def fit(self, key, opt, mean=None, scale=None, batch_size=8, niter=1000, nprint=10, monitor=None, retries=10):
        """
        Main function to fit a multivariate Gaussian distribution to the target

        Inputs:
          key: Random number generator key (jax.random.PRNGKey)
          mean : Optional, initial value of the mean. Expected None or array of size D
          cov : Optional, initial value of the covariance matrix. Expected None or array of size DxD
          batch_size : Optional, int. Number of samples to match scores for at every iteration
          niter : Optional, int. Total number of iterations
          nprint : Optional, int. Number of iterations after which to print logs
          monitor : Optional. Function to monitor the progress and track different statistics for diagnostics.
                    Function call should take the input tuple (iteration number, [mean, cov], lp, key, number of grad evals).
                    Example of monitor class is provided in utils/monitors.py
        Returns:
          mu : Array of shape D, fit of the mean
          cov : Array of shape DxD, fit of the covariance matrix
        """

        if self.jit_compile:    # I don't remember why we need to jit here. For monitor?
            lossf = jit(self.loss_function, static_argnums=(2))
        else:
            lossf = self.loss_function

        def opt_step(params, opt_state, key):
            loss, grads = jax.value_and_grad(lossf, argnums=0)(params, key, batch_size=batch_size)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        if self.jit_compile: opt_step = jit(opt_step)

        if mean is None:
            mean = jnp.zeros(self.D)
        if scale is None:
            scale  = np.ones(self.D)
            
        logscale = np.log(scale)
        params = (mean, logscale)

        # run optimization
        opt_state = opt.init(params)
        losses = []
        nevals = 1
        if monitor is not None:
            monitor.means = []
            monitor.psis = []
            monitor.llambdas = []
            monitor.iparams = []


        for i in range(niter + 1):
            if(i%(niter//nprint)==0):
                print(f'Iteration {i} of {niter}')
            if monitor is not None:
                if (i%monitor.checkpoint) == 0:
                    mean = params[0]
                    logscale = params[1]
                    psi = (np.exp(logscale))**2
                    llambda = np.zeros((self.D, 1))
                    monitor_lr(monitor, i, [mean, psi, llambda], self.lp, key, nevals=nevals)
                    #monitor(i, [mean, cov], self.lp, key, nevals=nevals)
                    nevals = 0

            j = 0
            while True:         # Sometimes run crashes due to a bad sample. Avoid that by re-trying.
                try:
                    key, _ = random.split(key)
                    params, opt_state, loss = opt_step(params, opt_state, key)
                    nevals += batch_size
                    break
                except Exception as e:
                    if j < retries :
                        j += 1
                        print(f"Failed with exception {e}")
                        print(f"Trying again {j} of {retries}")
                    else : raise e
            losses.append(loss)


        # Convert back to mean and covariance matrix
        mean = params[0]
        logscale = params[1]
        if monitor is not None:
            #monitor(i, [mean, cov], self.lp, key, nevals=nevals)
            psi = (np.exp(logscale))**2
            llambda = np.zeros((self.D, 1))
            monitor_lr(monitor, i, [mean, psi, llambda], self.lp, key, nevals=nevals)

        return mean, np.exp(logscale), losses



############
class BBVI():
    def __init__(self, D, lp, lp_g=None, jit_compile=True):
        """
        Inputs:
          D: (int) Dimensionality (number) of parameters.
          lp : Function to evaluate target log-probability distribution
               whose gradient can be evaluated with jax.grad(lp)
        """
        self.D = D
        self.lp = lp
        self.lp_g = lp_g
        self.idx_tril = jnp.stack(jnp.tril_indices(D)).T
        self.jit_compile = jit_compile
        if not jit_compile:
            print("Not using jit compilation. This may take longer than it needs to.")

    def minimize_loss(self, loss_function, key, opt, mean=None, cov=None, batch_size=8, niter=1000, nprint=10, monitor=None, retries=10):
        """
        Main function to fit a multivariate Gaussian distribution to the target

        Inputs:
          key: Random number generator key (jax.random.PRNGKey)
          mean : Optional, initial value of the mean. Expected None or array of size D
          cov : Optional, initial value of the covariance matrix. Expected None or array of size DxD
          batch_size : Optional, int. Number of samples to match scores for at every iteration
          niter : Optional, int. Total number of iterations
          nprint : Optional, int. Number of iterations after which to print logs
          monitor : Optional. Function to monitor the progress and track different statistics for diagnostics.
                    Function call should take the input tuple (iteration number, [mean, cov], lp, key, number of grad evals).
                    Example of monitor class is provided in utils/monitors.py
        Returns:
          mu : Array of shape D, fit of the mean
          cov : Array of shape DxD, fit of the covariance matrix
        """

        if self.jit_compile:    # I don't remember why we need to jit here. For monitor?
            lossf = jit(loss_function, static_argnums=(2))
        else:
            lossf = loss_function

        def opt_step(params, opt_state, key):
            loss, grads = jax.value_and_grad(lossf, argnums=0)(params, key, batch_size=batch_size)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        if self.jit_compile: opt_step = jit(opt_step)

        if mean is None:
            mean = jnp.zeros(self.D)
        if cov is None:
            cov = np.identity(self.D)

        # Optimization is done on unconstrained Cholesky factors of covariance matrix
        L = np.linalg.cholesky(cov)
        scales = jnp.array(L[np.tril_indices(self.D)])
        params = (mean, scales)

        # run optimization
        opt_state = opt.init(params)
        losses = []
        nevals = 1

        for i in range(niter + 1):
            if(i%(niter//nprint)==0):
                print(f'Iteration {i} of {niter}')
            if monitor is not None:
                if (i%monitor.checkpoint) == 0:
                    mean = params[0]
                    scales = params[1]
                    scale_tril = jnp.zeros((self.D, self.D))
                    scale_tril = scale_tril.at[self.idx_tril[:, 0], self.idx_tril[:, 1]].set(scales)
                    cov = np.matmul(scale_tril, scale_tril.T)
                    monitor(i, [mean, cov], self.lp, key, nevals=nevals)
                    nevals = 0

            j = 0
            while True:         # Sometimes run crashes due to a bad sample. Avoid that by re-trying.
                try:
                    key, _ = random.split(key)
                    params, opt_state, loss = opt_step(params, opt_state, key)
                    nevals += batch_size
                    break
                except Exception as e:
                    if j < retries :
                        j += 1
                        print(f"Failed with exception {e}")
                        print(f"Trying again {j} of {retries}")
                    else : raise e
            losses.append(loss)


        # Convert back to mean and covariance matrix
        mean = params[0]
        scales = params[1]
        scale_tril = jnp.zeros((self.D, self.D))
        scale_tril = scale_tril.at[self.idx_tril[:, 0], self.idx_tril[:, 1]].set(scales)
        cov = np.matmul(scale_tril, scale_tril.T)
        if monitor is not None:
            monitor(i, [mean, cov], self.lp, key, nevals=nevals)

        return mean, cov, losses


################################################################################
class ADVI(BBVI):
    """
    Class for fitting a multivariate Gaussian distribution with dense covariance matrix
    by maximizing ELBO.
    """

    def __init__(self, D, lp, jit_compile=True):
        """
        Inputs:
          D: (int) Dimensionality (number) of parameters.
          lp : Function to evaluate target log-probability distribution
               whose gradient can be evaluated with jax.grad(lp)
        """
        super().__init__(D=D, lp=lp, jit_compile=jit_compile)

    def loss_function(self, params, key, batch_size):
        """
        Internal function to evaluate negative-ELBO which is the loss function
        """
        loc, scales = params
        scale_tril = jnp.zeros((self.D, self.D))
        scale_tril = scale_tril.at[self.idx_tril[:, 0], self.idx_tril[:, 1]].set(scales)
        q = MultivariateNormal(loc=loc, scale_tril=scale_tril)
        #
        samples = q.sample(key, (batch_size,))
        logl = jnp.sum(self.lp(samples))
        logq = jnp.sum(q.log_prob(samples))
        elbo = logl - logq
        negelbo = -1. * elbo
        return negelbo

    def fit(self, key, opt, mean=None, cov=None, batch_size=8, niter=1000, nprint=10, monitor=None, retries=10):
        """
        Main function to fit a multivariate Gaussian distribution to the target

        Inputs:
          key: Random number generator key (jax.random.PRNGKey)
          mean : Optional, initial value of the mean. Expected None or array of size D
          cov : Optional, initial value of the covariance matrix. Expected None or array of size DxD
          batch_size : Optional, int. Number of samples to match scores for at every iteration
          niter : Optional, int. Total number of iterations
          nprint : Optional, int. Number of iterations after which to print logs
          monitor : Optional. Function to monitor the progress and track different statistics for diagnostics.
                    Function call should take the input tuple (iteration number, [mean, cov], lp, key, number of grad evals).
                    Example of monitor class is provided in utils/monitors.py
        Returns:
          mu : Array of shape D, fit of the mean
          cov : Array of shape DxD, fit of the covariance matrix        """

        return self.minimize_loss(self.loss_function, key, opt, mean=mean, cov=cov, batch_size=batch_size, niter=niter, nprint=nprint, monitor=monitor, retries=retries)


################################################################################
class Scorenorm(BBVI):
    """
    Class for fitting a multivariate Gaussian distribution with dense covariance matrix
    by minimizing the scorenorm.
    """

    def __init__(self, D, lp, lp_g, jit_compile=True):
        """
        Inputs:
          D: (int) Dimensionality (number) of parameters.
          lp : Function to evaluate target log-probability distribution
               whose gradient can be evaluated with jax.grad(lp)
        """
        super().__init__(D=D, lp=lp, lp_g=lp_g, jit_compile=jit_compile)

    def loss_function(self, params, key, batch_size):
        """
        Internal function to evaluate the L2 loss between the score of the true and the variational distribution.
        """
        loc, scales = params
        scale_tril = jnp.zeros((self.D, self.D))
        scale_tril = scale_tril.at[self.idx_tril[:, 0], self.idx_tril[:, 1]].set(scales)
        q = MultivariateNormal(loc=loc, scale_tril=scale_tril)
        q_lp = lambda x: jnp.sum(q.log_prob(x))
        #q_lp_g = jit(grad(q_lp, argnums=0))
        q_lp_g = grad(q_lp, argnums=0)
        #
        samples = q.sample(key, (batch_size,))
        true_score = self.lp_g(samples)
        var_score = q_lp_g(samples)
        #scorediff = scale_tril.T.dot(true_score - var_score)
        scorediff = (true_score - var_score).dot(scale_tril)
        #print(scorediff.shape())
        scorenorm = jnp.mean(jnp.sum(scorediff**2, axis=1)**0.5, axis=0)
        return scorenorm


    def fit(self, key, opt, mean=None, cov=None, batch_size=8, niter=1000, nprint=10, monitor=None, retries=10):
        """
        Main function to fit a multivariate Gaussian distribution to the target

        Inputs:
          key: Random number generator key (jax.random.PRNGKey)
          mean : Optional, initial value of the mean. Expected None or array of size D
          cov : Optional, initial value of the covariance matrix. Expected None or array of size DxD
          batch_size : Optional, int. Number of samples to match scores for at every iteration
          niter : Optional, int. Total number of iterations
          nprint : Optional, int. Number of iterations after which to print logs
          monitor : Optional. Function to monitor the progress and track different statistics for diagnostics.
                    Function call should take the input tuple (iteration number, [mean, cov], lp, key, number of grad evals).
                    Example of monitor class is provided in utils/monitors.py
        Returns:
          mu : Array of shape D, fit of the mean
          cov : Array of shape DxD, fit of the covariance matrix
        """

        return self.minimize_loss(self.loss_function, key, opt, mean=mean, cov=cov, batch_size=batch_size, niter=niter, nprint=nprint, monitor=monitor, retries=retries)

    

class Fishernorm(BBVI):
    """
    Class for fitting a multivariate Gaussian distribution with dense covariance matrix
    by minimizing the scorenorm.
    """

    def __init__(self, D, lp, lp_g, jit_compile=True):
        """
        Inputs:
          D: (int) Dimensionality (number) of parameters.
          lp : Function to evaluate target log-probability distribution
               whose gradient can be evaluated with jax.grad(lp)
        """
        super().__init__(D=D, lp=lp, lp_g=lp_g, jit_compile=jit_compile)

    def loss_function(self, params, key, batch_size):
        """
        Internal function to evaluate the L2 loss between the score of the true and the variational distribution.
        """
        loc, scales = params
        scale_tril = jnp.zeros((self.D, self.D))
        scale_tril = scale_tril.at[self.idx_tril[:, 0], self.idx_tril[:, 1]].set(scales)
        q = MultivariateNormal(loc=loc, scale_tril=scale_tril)
        q_lp = lambda x: jnp.sum(q.log_prob(x))
        #q_lp_g = jit(grad(q_lp, argnums=0))
        q_lp_g = grad(q_lp, argnums=0)
        #
        samples = q.sample(key, (batch_size,))
        true_score = self.lp_g(samples)
        var_score = q_lp_g(samples)
        scorediff = true_score - var_score
        scorenorm = jnp.mean(jnp.sum(scorediff**2, axis=1)**0.5, axis=0)
        return scorenorm


    def fit(self, key, opt, mean=None, cov=None, batch_size=8, niter=1000, nprint=10, monitor=None, retries=10):
        """
        Main function to fit a multivariate Gaussian distribution to the target

        Inputs:
          key: Random number generator key (jax.random.PRNGKey)
          mean : Optional, initial value of the mean. Expected None or array of size D
          cov : Optional, initial value of the covariance matrix. Expected None or array of size DxD
          batch_size : Optional, int. Number of samples to match scores for at every iteration
          niter : Optional, int. Total number of iterations
          nprint : Optional, int. Number of iterations after which to print logs
          monitor : Optional. Function to monitor the progress and track different statistics for diagnostics.
                    Function call should take the input tuple (iteration number, [mean, cov], lp, key, number of grad evals).
                    Example of monitor class is provided in utils/monitors.py
        Returns:
          mu : Array of shape D, fit of the mean
          cov : Array of shape DxD, fit of the covariance matrix
        """

        return self.minimize_loss(self.loss_function, key, opt, mean=mean, cov=cov, batch_size=batch_size, niter=niter, nprint=nprint, monitor=monitor, retries=retries)




################################################################################
class ADVI_LR(BBVI):
    """
    Class for fitting a multivariate Gaussian distribution with dense covariance matrix
    by maximizing ELBO.
    """

    def __init__(self, D, rank, lp, jit_compile=False):
        """
        Inputs:
          D: (int) Dimensionality (number) of parameters.
          lp : Function to evaluate target log-probability distribution
               whose gradient can be evaluated with jax.grad(lp)
        """
        super().__init__(D=D, lp=lp, jit_compile=jit_compile)
        print('jit compile', jit_compile, self.jit_compile)
        self.rank = rank
        

    def loss_function(self, params, key, batch_size):
        """
        Internal function to evaluate negative-ELBO which is the loss function
        """
        loc, psi, llambda = params
        K = llambda.shape[1]
        key, key_eps, key_z = random.split(key, 3)

        eps = Normal(0, 1).sample(key_eps, (batch_size, self.D))
        z = Normal(0, 1).sample(key_z, (batch_size, K))
        samples = loc + psi**0.5 * eps + (llambda@z.T).T
        logl = jnp.sum(self.lp(samples))
        logq = jnp.sum(logp_lr_vmap(samples, loc, psi, llambda))
        elbo = logl - logq
        negelbo = -1. * elbo
        return negelbo

    
    def fit(self, key, opt, mean=None, psi=None, llambda=None, batch_size=8, niter=1000, nprint=10, monitor=None, retries=10, jitter=1e-6):
        """
        """

        if self.jit_compile:    # I don't remember why we need to jit here. For monitor?
            lossf = jit(self.loss_function, static_argnums=(2))
        else:
            lossf = self.loss_function

        def opt_step(params, opt_state, key):
            loss, grads = jax.value_and_grad(lossf, argnums=0)(params, key, batch_size=batch_size)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        if self.jit_compile: opt_step = jit(opt_step)

        # Optimization is done on unconstrained Cholesky factors of covariance matrix
        K = self.rank
        if mean is None:
            mean = jnp.zeros(self.D)
        if llambda is None:
            llambda = np.random.normal(0, 1, size=(self.D, K)) 
        if psi is None:
            psi = np.random.random(self.D)

        params = (mean, psi, llambda)

        # run optimization
        opt_state = opt.init(params)
        losses = []
        nevals = 1
        if monitor is not None:
            monitor.means = []
            monitor.psis = []
            monitor.llambdas = []
            monitor.iparams = []

            
        for i in range(niter + 1):
            if(i%(niter//nprint)==0):
                print(f'Iteration {i} of {niter}')
            if monitor is not None:
                if (i%monitor.checkpoint) == 0:
                    monitor_lr(monitor, i, [mean, psi, llambda], self.lp, key, nevals=nevals)
                    nevals = 0

            j = 0
            while True:         # Sometimes run crashes due to a bad sample. Avoid that by re-trying.
                try:
                    key, _ = random.split(key)
                    params, opt_state, loss = opt_step(params, opt_state, key)
                    mean, psi, llambda = params
                    #psi = psi + jitter
                    psi = jnp.maximum(psi, jitter)

                    params = (mean, psi, llambda)
                    nevals += batch_size
                    break
                except Exception as e:
                    if j < retries :
                        j += 1
                        print(f"Failed with exception {e}")
                        print(f"Trying again {j} of {retries}")
                    else : raise e
            losses.append(loss)


        if monitor is not None:
            monitor_lr(monitor, i, [mean, psi, llambda], self.lp, key, nevals=nevals, force_save=True)
            monitor.means.append(mean)
            monitor.psis.append(psi)
            monitor.llambdas.append(llambda)

        return mean, psi, llambda, losses

