



from flax import nnx
import jax.numpy as jnp
import jax





class RMSNorm(nnx.Module): 
    def __init__(self, hidden_size: int, eps: float = 1e-9):
        super().__init__()
        self.weights = nnx.Param(jnp.ones((hidden_size, ), dtype=jnp.float32))
        self.eps = eps

    def __call__(self, hidden_states: jax.Array):
        dtype = hidden_states.dtype
        hidden_states_f32 = hidden_states.astype(jnp.float32)
        variance = jnp.mean(jnp.pow(hidden_states_f32, 2), axis=-1, keepdims=True)
        hidden_states = hidden_states_f32 * jax.lax.rsqrt(variance + self.eps)
        return (hidden_states * self.weights).astype(dtype=dtype)