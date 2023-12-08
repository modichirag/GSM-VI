# GSM-VI
### Code for Variational Inference (VI) with Gaussian Score Matching (GSM).

This repo implements code for paper https://arxiv.org/abs/2307.07849.

GSM-VI fits a multivariate Gasussian distribution with dense covaraince matrix to the target distribution
by score matching. It only requires access to the score function i.e. the gradient of the target log-probability
distribution and implements analytic updates for the variational parameters (mean and covariance matrix).

The algorithm is implemented in Jax for performance reasons like jit compilation,
but it does not rely on this backend as long as one can provide the score function.

For comaprison, we also provide implementation of ADVI algorithm (https://arxiv.org/abs/1603.00788),
which is another common variants of fitting a multivariate Gaussian distribution with VI.

### Code Dependencies: <br>
The code is written in python3 and uses the basic python packages like numpy and matplotlib.<br>
GSM algorithm is implemented in `Jax` to benefit from jit-compilation.<br>

#### Optional dependencies 
These will not be installed with the package but are needed to run the `examples` and `advi`.

The target distributions in example files are implemented in `numpyro`.<br>
For using other algorithms and utilities, following optional packages are requied: <br>
ADVI uses `optax` for maximizing ELBO.<br>
LBFGS initialization for initializing variational distributions uses `scipy`. 

### Other utilities :<br>
We provide LBFGS initilization for the variational distribution which can be used with GSM and ADVI. <br>
We also provide a Monitor class to monitor the KL divergence over iterations as the algorithms progress.


### Basic usage

The following is the minimal code to use GSM to fit the parameters `x` of a `model` which gives access to `log_prob` function. 
```
dimensions = D
model =  setup_model(D=D)
lp = jit(lambda x: jnp.sum(model.log_prob(x)))
lp_g = jit(grad(lp, argnums=0))

gsm = GSM(D=D, lp=lp, lp_g=lp_g)
mean_fit, cov_fit = gsm.fit(key=random.PRNGKey(99), niter=500)
```

### Starting point :<br>
We provide simple examples in `examples/` folder to fit a target multivariate Gaussian distribution with GSM and ADVI. <br>
```
cd examples
python3 example_gsm.py
python3 example_advi.py
```
An example on how to use the Monitor class and LBFGS initialization is in `examples/example_initializers.py`
```
cd examples
python3 example_initializers.py
```
