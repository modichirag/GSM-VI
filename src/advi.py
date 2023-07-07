import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad, random
from numpyro.distributions import MultivariateNormal
import optax
      
class ADVI():
    
    def __init__(self, D, lp):
        self.D = D
        self.lp = lp
        self.idx_tril = jnp.stack(jnp.tril_indices(D)).T
        
    def neg_elbo(self, params, key, batch_size):

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

    def fit(self, key, opt, mu=None, cov=None, batch_size=8, niter=1000, nprint=10):
    
    
        lossf = jit(self.neg_elbo, static_argnums=(2))

        @jit
        def opt_step(params, opt_state, key):
            loss, grads = jax.value_and_grad(lossf, argnums=0)(params, key, batch_size=batch_size)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        if mu is None:
            mu = jnp.zeros(self.D)
        if cov is None:
            cov = np.identity(self.D)
        L = np.linalg.cholesky(cov)
        scales = jnp.array(L[np.tril_indices(self.D)])
            
        params = (mu, scales)

        # run optimization
        opt_state = opt.init(params)
        losses = []

        for i in range(niter + 1):
            if(i%(niter//nprint)==0):
                print(f'Iteration {j} of {niter}')
            params, opt_state, loss = opt_step(params, opt_state, key)
            key, _ = random.split(key)
            losses.append(loss)

            if monitor is not None:
                monitor(i, [mu, cov], self.lp, key)

        # Convert back to mean and covariance matrix
        mu = params[0]
        scales = params[1]
        scale_tril = jnp.zeros((self.D, self.D))
        scale_tril = scale_tril.at[self.idx_tril[:, 0], self.idx_tril[:, 1]].set(scales)
        cov = np.matmul(scale_tril, scale_tril.T)
        return mu, cov, losses
