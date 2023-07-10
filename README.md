# GSM-VI
### Code for Gaussian Score Matching Variational Inference

This repo implements the Gaussian Score Matching (GSM) algorithm for Variational Inference (VI)
which fits a multivariate Gasussian distribution with dense covaraince matrix to the target distribution
by score matching. It only requires access to the score function i.e. the gradient of the target log-probability
distribution and implements analytic updates for the variational parameters (mean and covariance matrix).

The algorithm by implemented in Jax for performance reasons like jit compilation,
but it does not rely on this backend as long as one can provide the score function.
To re-inforce this, we also provide a pure numpy implementation in src/gsm_numpy.py,
and this can be adapted to other backends.

For comaprison, we also provide implementations of two other common variants of fitting a
multivariate Gaussian distribution with VI - ADVI (https://arxiv.org/abs/1603.00788)
and Pathfinder (https://arxiv.org/abs/2108.03782). <br>


Dependencies: <br>
The algorithms are implemented in Jax.  
The target distributions in example files are implemented in numpyro 
ADVI uses optax for maximizing ELBO. 
Pathfinder uses the implementation in Blackjax which needs to be installed.


Starting point :<br>
We provide a simple example in examples/ folder to fit a target multivariate Gaussian distribution
with all three algorithms. <br>

Other utilities :<br>
We provide LBFGS initilization for GSM and ADVI, as well as a Monitor class to monitor the KL divergence
over iterations as the algorithms progress. An example on how to use this is in examples/example_gsm_initializers.py
