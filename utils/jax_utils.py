import jax.numpy as jnp
from jax import custom_jvp


@custom_jvp
def lp(x):
    result_shape = jax.ShapeDtypeStruct((), x.dtype)
    return jax.pure_callback(lambda s: np.array(model.lp(np.array(s))), result_shape, x)

@lp.defjvp
def lp_jvp(primals, tangents):
    x = primals[0]
    x_dot = tangents
    result_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    primal_out = lp(x)
    tangent_out = jax.pure_callback(lambda s: np.array(model.lp_g(np.array(s))[1]), result_shape, x)
    tangent_out = jnp.matmul(tangent_out, tangents[0])
    return primal_out, tangent_out


