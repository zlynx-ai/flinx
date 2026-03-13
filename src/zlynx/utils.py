
import jax, jax.numpy as jnp

def get_act_fn(act):
    if callable(act) or not isinstance(act, str):
        return act
    return getattr(jax.nn, act)


def get_dtype(dtype):
    if callable(dtype) or not isinstance(dtype, str):
        return dtype
    return getattr(jnp, dtype)