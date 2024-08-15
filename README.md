# GSM-VI: Gaussian score-based variational inference algorithms

This repository provides a Python implementation of two score-based (black-box)
variational inference algorithms:
1. GSM-VI, which is described in the NeurIPS 2023 paper https://arxiv.org/abs/2307.07849.
2. Batch and match (BaM), which is described in the ICML 2024 paper https://arxiv.org/pdf/2402.14758.

We describe each of these in the following two sections.

## Variational Inference (VI) with Gaussian Score Matching (GSM) (NeurIPS 2023)

GSM-VI fits a multivariate Gasussian distribution with dense covaraince matrix to the target distribution
by score matching. It only requires access to the score function i.e. the gradient of the target log-probability
distribution and implements analytic updates for the variational parameters (mean and covariance matrix).

### Installation: <br>
The code is available on `PyPI`
```
pip install gsmvi
```

### Usage
The simplest version of the algorithm is written in numpy.
The following is the minimal code to use GSM to fit the parameters `x` of a `model` given its `log_prob` and `log_prob_grad` functions.
See `example/example_gsm_numpy.py` for a full example.
```
dimensions = D
def log_prob(x):
  # return log_prbability at sample x
  ...

def log_prob_grad(x):
  # return the score fuction i.e. the gradient of log_prbability at sample x
  ...

from gsmvi.gsm_numpy import GSM
gsm = GSM(D=D, lp=log_prob, lp_g=log_prob_grad)
random_seed = 99
number_of_iterations = 500
mean_fit, cov_fit = gsm.fit(key=random_seed, niter=number_of_iterations)
```

A more efficient version of the algorithm is implemented in Jax where it can benefit from jit compilation. The basic signature stays the same.
See `example/example_gsm.py` for a full example.
```
dimensions = D
model =  setup_model(D=D)     # Ths example sets up a numpyro model which has log_prob attribute implemented
lp = jit(lambda x: jnp.sum(model.log_prob(x)))
lp_g = jit(grad(lp, argnums=0))

from gsmvi.gsm import GSM
gsm = GSM(D=D, lp=lp, lp_g=lp_g)
mean_fit, cov_fit = gsm.fit(key=random.PRNGKey(99), niter=500)
```

#### Other utilities:<br>
- For comparison, we also provide implementation of ADVI algorithm (https://arxiv.org/abs/1603.00788),
another common approach to fit a multivariate Gaussian variational distribution which maximizes ELBO.
- We provide LBFGS initilization for the variational distribution which can be used with GSM and ADVI.
- We also provide a Monitor class to monitor the KL divergence over iterations as the algorithms progress.

### Code Dependencies<br>
The vanilla code is written in python3 and does not have any dependencies. <br>

#### Optional dependencies
These will not be installed with the package and should be installed by user depending on the use-case.

The Jax version of the code requires `jax` and `jaxlib`.<br>
The target distributions in example files other than example_gsm_numpy.py are implemented in `numpyro`.<br>
ADVI algorithm uses `optax` for maximizing ELBO.<br>
LBFGS initialization for initializing variational distributions uses `scipy`.

### Starting point<br>
We provide simple examples in `examples/` folder to fit a target multivariate Gaussian distribution with GSM and ADVI. <br>
```
cd examples
python3 example_gsm_numpy.py  # vanilla example in numpy, no dependencies
python3 example_gsm.py        # jax version, requires jax and numpyro
python3 example_advi.py       # jax version, requires jax, numpyro and optax
```
An example on how to use the Monitor class and LBFGS initialization is in `examples/example_initializers.py`
```
cd examples
python3 example_initializers.py   # jax version, requires jax, numpyro, optax and scipy
```

## Batch and match: black-box variational inference with a score-based divergence (ICML 2024)

[Batch and match (BaM)](https://arxiv.org/pdf/2402.14758) also fits a full covariance multivariate Gaussian and recovers (a version of) GSM as a special case.
In the BaM algorithm, a score-based divergence is minimized.
The code is set up similarly to the GSM code. Currently, it is not yet available
on `PyPI`.

To install, run
```
pip install -e .
```


The example usage code is in `examples/example_bam.py':

```
cd examples
python3 example_bam.py        # jax version, requires jax and numpyro
```

Note that this installation approach also includes the GSM and ADVI code
examples above.

