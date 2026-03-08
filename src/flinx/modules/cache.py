





import jax
import jax.numpy as jnp
from flax import nnx


class KVCacheState(nnx.Variable):
    pass


class CacheIndex(nnx.Variable):
    pass


class KVCacheBase(nnx.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        pass

    def init_cache_state(self, batch_size: int, max_seq_len: int):
        if (
            hasattr(self, "kv_head")
            and hasattr(self, "head_dim")
            and hasattr(self, "dtype")
        ):
            self.k_cache = KVCacheState(
                jnp.zeros(
                    (batch_size, max_seq_len, self.kv_head, self.head_dim),
                    dtype=self.dtype,
                )
            )
            self.v_cache = KVCacheState(
                jnp.zeros(
                    (batch_size, max_seq_len, self.kv_head, self.head_dim),
                    dtype=self.dtype,
                )
            )
            self.cache_index = CacheIndex(jnp.zeros((), dtype=jnp.int32))

    def update_cache(self, key: jax.Array, value: jax.Array):
        if not hasattr(self, "k_cache") or self.k_cache is None:
            return key, value

        key = key.astype(self.k_cache.value.dtype)
        value = value.astype(self.v_cache.value.dtype)

        S = key.shape[1]

        self.k_cache.value = jax.lax.dynamic_update_slice(
            self.k_cache.value, key, (0, self.cache_index.value, 0, 0)
        )
        self.v_cache.value = jax.lax.dynamic_update_slice(
            self.v_cache.value, value, (0, self.cache_index.value, 0, 0)
        )
        self.cache_index.value += S

        return self.k_cache.value, self.v_cache.value
