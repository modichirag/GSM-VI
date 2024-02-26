# GSM-VI
### Code for Gaussian Score Matching (GSM) VI and "batch and match" (BaM) VI

This repo implements code for the methods:
1. Gaussian Score Matching VI (GSM-VI). NeurIPS 2023. https://arxiv.org/abs/2307.07849.
2. Batch and match (BaM): black-box variational inference using a score-based divergence.

Both GSM-VI and BaM fit a multivariate Gaussian distribution with a dense covariance matrix to the target distribution
by score matching. The methods only require access to the score function, i.e., the gradient of the target log-probability
distribution and implements analytic updates for the variational parameters (mean and covariance matrix).

The algorithms are implemented in JAX for performance reasons like jit compilation,
but it does not rely on this backend as long as one can provide the score function.
To re-inforce this, we also provide a pure numpy implementation for GSM in `src/gsm_numpy.py`,
and this can be adapted to other backends.

For comparison, we also provide implementations of two other common variants of fitting a
multivariate Gaussian distribution with VI:<br>
ADVI (https://arxiv.org/abs/1603.00788)
and Pathfinder (https://arxiv.org/abs/2108.03782). <br>


### Code Dependencies: <br>
The code is written in python3 and uses the basic python packages like numpy and matplotlib.<br>
GSM algorithm is implemented in `Jax` to benefit from jit-compilation.<br>
The target distributions in example files are implemented in `numpyro`.<br>

#### Optional dependencies
For using other algorithms and utilities, following optional packages are requied: <br>
ADVI uses `optax` for maximizing ELBO.<br>
Pathfinder uses the implementation in `Blackjax` which needs to be installed.<br>
LBFGS initialization for initializing variational distributions uses `scipy`.


### Starting point :<br>
We provide simple examples in `examples/` folder to fit a target multivariate Gaussian distribution
with all three algorithms. <br>
```
cd examples
python3 example_gsm.py
python3 example_bam.py
python3 example_advi.py
python3 example_pathfinder.py
```

### Other utilities :<br>
We provide LBFGS initilization for the variational distribution which can be used with GSM and ADVI. <br>
We also provide a Monitor class to monitor the KL divergence over iterations as the algorithms progress.

An example on how to use both these is in `examples/example_gsm_initializers.py`
