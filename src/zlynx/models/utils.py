
import jax, jax.numpy as jnp

def get_act_fn(act):
    if callable(act):
        return act
    return getattr(jax.nn, act)


def get_dtype(dtype):
    if callable(dtype):
        return dtype
    return getattr(jnp, dtype)