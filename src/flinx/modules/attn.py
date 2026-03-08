

import jax, jax.numpy as jnp
from flax import nnx

from .cache import KVCacheBase
from .rope import apply_rope

class Attention(KVCacheBase):
    def __init__(
        self, key, 
        hidden_size: int, 
        attention_head: int, 
        head_dim: int, 
        kv_head: int | None = None, 
        bias: bool = False, 
        layer_idx: int | None = None,
        dtype=jnp.bfloat16,
        use_cache: bool = True
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.dtype = dtype
        self.use_cache = use_cache

        self.attention_head = attention_head
        kv_head = kv_head if kv_head is not None else attention_head
        self.head_dim = head_dim
        self.kv_head = kv_head

        q_key, k_key, v_key, o_key = jax.random.split(key, 4)
        self.q_proj = nnx.Linear(
            hidden_size, 
            attention_head * head_dim,
            use_bias=bias, 
            dtype=dtype, 
            param_dtype=dtype, 
            rngs=nnx.Rngs(q_key)
        )
        self.k_proj = nnx.Linear(
            hidden_size, 
            kv_head * head_dim,
            use_bias=bias, 
            dtype=dtype, 
            param_dtype=dtype, 
            rngs=nnx.Rngs(k_key)
        )
        self.v_proj = nnx.Linear(
            hidden_size, 
            kv_head * head_dim,
            use_bias=bias, 
            dtype=dtype, 
            param_dtype=dtype, 
            rngs=nnx.Rngs(v_key)
        )
        self.o_proj = nnx.Linear(
            # GQA
            attention_head * head_dim \
                if attention_head >= kv_head else kv_head * head_dim,  # MQA
            hidden_size,
            use_bias=bias, 
            dtype=dtype, 
            param_dtype=dtype, 
            rngs=nnx.Rngs(o_key)
        )

    def __call__(
        self, hidden_states: jax.Array, 
        attention_mask: jax.Array | None = None, 
        position_embedding: tuple[jax.Array] | None = None
    ):
        input_shape = hidden_states.shape[:-1]
        query = self.q_proj(hidden_states).reshape(*hidden_states.shape[:-1], -1, self.head_dim)
        key = self.k_proj(hidden_states).reshape(*hidden_states.shape[:-1], -1, self.head_dim)
        value = self.v_proj(hidden_states).reshape(*hidden_states.shape[:-1], -1, self.head_dim)

        if position_embedding is not None:
            cos, sin = position_embedding
            query, key = apply_rope(query, key, cos, sin)

        if self.use_cache:
            key, value = self.update_cache(key, value)

        # GQA: repeat KV heads to match query heads
        if self.attention_head > self.kv_head:
            num_groups = self.attention_head // self.kv_head
            key = jnp.repeat(key, num_groups, axis=-2)
            value = jnp.repeat(value, num_groups, axis=-2)

        if attention_mask is not None and key.shape[1] > attention_mask.shape[-1]:
            pad_len = key.shape[1] - attention_mask.shape[-1]
            attention_mask = jnp.pad(attention_mask, ((0,0), (0,0), (0,0), (0, pad_len)))

        hidden_states = nnx.dot_product_attention(
            query, key, value, mask=attention_mask
        )

        hidden_states = hidden_states.reshape(*input_shape, -1)
        return self.o_proj(hidden_states)