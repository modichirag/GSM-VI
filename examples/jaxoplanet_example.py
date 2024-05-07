import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import arviz as az
import numpyro
import jaxoplanet
from jaxoplanet.light_curves import limb_dark_light_curve
from jaxoplanet.orbits import TransitOrbit

from numpyro import distributions
from jax import jit, grad
import tensorflow_probability as tfp
from tensorflow_probability.substrates.jax import bijectors as bj
import numpyro_ext

import sys
sys.path.append('../gsmvi/')
from gsm import GSM

float32 = np.float32


########
# Generate DATA
# Simulate some data with Gaussian noise
D = 7
random = np.random.default_rng(42)
PERIOD = random.uniform(2, 5)  # day
T0 = PERIOD * random.uniform()  # day
DURATION = 0.5  # day
B = 0.5  # impact parameter
ROR = 0.08  # planet radius / star radius
U = np.array([0.1, 0.06])  # limb darkening coefficients
yerr = 5e-4  # flux uncertainty
t = np.arange(0, 17, 0.05)  # day


orbit = TransitOrbit(
    period=PERIOD, duration=DURATION, time_transit=T0, impact_param=B, radius=ROR
)
y_true = limb_dark_light_curve(orbit, U)(t)
y = y_true + yerr * random.normal(size=len(t))


################
def constrain_bounded_bijector(a, b):
    shift = a
    scale = b-a
    transform = bj.Chain([bj.Shift(shift=shift), bj.Scale(scale=scale), bj.Sigmoid()])
    return transform


class JaxPlanet():
    
    def __init__(self, yobs, t, yerr, 
                T0, PERIOD, DURATION,
                radius_limits=[0.01, 0.2], 
                rawb_limits=[0.0, 1.], 
                q_limits=[0., 1.]):
        
        self.yobs = yobs
        self.t = t
        self.yerr = yerr
        self.T0, self.PERIOD, self.DURATION = T0, PERIOD, DURATION

        self.transform_radius = constrain_bounded_bijector(radius_limits[0], radius_limits[1])
        self.transform_rawb = constrain_bounded_bijector(rawb_limits[0], rawb_limits[1])
        self.transform_q = constrain_bounded_bijector(q_limits[0], q_limits[1])
        self.dist_radius = distributions.Uniform(radius_limits[0], radius_limits[1])
        self.dist_rawb = distributions.Uniform(rawb_limits[0], rawb_limits[1])
        self.dist_q = distributions.Uniform(q_limits[0], q_limits[1])
        
    def constrain_period(self, log_period):
        return jnp.exp(log_period)
    
    def constrain_duration(self, log_duration):
        return jnp.exp(log_duration)
    
    def constrain_radius(self, radius_u):
        radius = self.transform_radius(radius_u)
        return radius
    
    def constrain_rawb(self, rawb_u):
        rawb = self.transform_rawb(rawb_u)
        return rawb

    def constrain_q(self, q_u):
        q = self.transform_q(q_u)
        return q

    def transform_parameters(self, params):
        assert len(params.shape) == 2
        t0_u, period_u, duration_u, r_u, rawb_u, q0_u, q1_u = params
        t0 = t0_u 
        period = self.constrain_period(period_u)
        duration = self.constrain_duration(duration_u)
        r = self.constrain_radius(r_u)
        rawb = self.constrain_rawb(rawb_u)
        b = rawb * (1 + r)
        q0 = self.constrain_q(q0_u)
        q1 = self.constrain_q(q1_u)
        q = jnp.stack([q0, q1]).T
        u = numpyro_ext.distributions.QuadLDTransform()(q)
        # return t0, period, duration, r, b, u[0], u[1]
        return jnp.array([t0, period, duration, r, b, u[..., 0], u[..., 1]]).T

    
    def log_prob_model(self, params):
        
        t0_u, period_u, duration_u, r_u, rawb_u, q0_u, q1_u = params
                    
        log_prior = 0. 
        log_prior += distributions.Normal(self.T0, 1).log_prob(t0_u)
        log_prior += distributions.Normal(jnp.log(self.PERIOD), 0.1).log_prob(period_u)
        log_prior += distributions.Normal(jnp.log(self.DURATION), 0.1).log_prob(duration_u)

        r = self.constrain_radius(r_u)
        lp_r = self.dist_radius.log_prob(r)
        ldj_r = self.transform_radius.forward_log_det_jacobian(r_u)
        log_prior += lp_r + ldj_r
        
        rawb = self.constrain_rawb(rawb_u)
        b = rawb * (1 + r)
        lp_rawb = self.dist_radius.log_prob(rawb)
        ldj_rawb = self.transform_radius.forward_log_det_jacobian(rawb_u)
        log_prior += lp_rawb + ldj_rawb
        
        q0 = self.constrain_q(q0_u)
        q1 = self.constrain_q(q1_u)
        q = jnp.stack([q0, q1])
        u = numpyro_ext.distributions.QuadLDTransform()(q)
        lp_u = numpyro_ext.distributions.QuadLDParams().log_prob(u)
        ldj_q = self.transform_q.forward_log_det_jacobian(q0_u) + self.transform_q.forward_log_det_jacobian(q1_u)
        ldj_u = numpyro_ext.distributions.QuadLDTransform().log_abs_det_jacobian(q, 0.)
        log_prior += lp_u + ldj_q + ldj_u 
 
        # return log_prior

        # The orbit and light curve
        period = self.constrain_period(period_u)
        duration = self.constrain_duration(duration_u)
        t0 = t0_u*1.
                                       
        orbit = TransitOrbit(
            period=period, duration=duration, time_transit=t0, impact_param=b, radius=r
        )
        y_pred = limb_dark_light_curve(orbit, u)(self.t)
        log_likelihood = jnp.sum(distributions.Normal(y_pred, self.yerr).log_prob(self.yobs))

        log_prob = log_prior + log_likelihood

        return log_prob


planet = JaxPlanet(y.astype(float32), np.float32(t), np.float32(yerr), 
               T0=np.float32(T0), PERIOD=np.float32(PERIOD), DURATION=np.float32(DURATION))
truths= jnp.array([T0, PERIOD, DURATION, ROR, B, U[0], U[1]], dtype=float32)
q0 = [T0, 
      np.log(PERIOD), 
      np.log(DURATION),
      planet.transform_radius.inverse(ROR),
      planet.transform_rawb.inverse(B/(1 + ROR)),
      *planet.transform_q.inverse(numpyro_ext.distributions.QuadLDTransform().inv(truths[-2:]))]
q0 = jnp.array(q0)
print('initial point : ', q0)

lp = jax.jit(planet.log_prob_model)
lp_g = jax.jit(jax.grad(planet.log_prob_model))
lp_vmap = lambda x: jax.vmap(lp, in_axes=0)(x.astype(float32))
lp_g_vmap = lambda x: jax.vmap(lp_g, in_axes=0)(x.astype(float32))

print('log prob at initial point : ', lp(q0))
print('grad : ', lp_g(q0))

# alg = GSM(D, lp, lp_g)
alg = GSM(D, lp_vmap, lp_g_vmap)
key = jax.random.PRNGKey(2)
mean, cov = alg.fit(key, mean=q0, cov=np.eye(D).astype(float32)*1e-3, batch_size=4, niter=1000)
print('fit mean : ', mean)
print('transformed params : ', planet.transform_parameters(mean.reshape([1, -1]).T))
print('true value : ', truths)
