


import jax
import jax.numpy as jnp
from flax import struct
from typing_extensions import Callable

from ..config import LanguageConfig

@struct.dataclass
class LlamaConfig(LanguageConfig):
    architecture: str = "LlamaLanguageModel"
    vocab_size: int = 80000
    hidden_size: int = 1024
    intermediate_size: int = 2048
    act_fn: str = "silu"
    num_hidden_layers: int = 4
    norm_eps: float = 1e-6
    bias: bool = False
    dtype: str = "bfloat16"
    param_dtype: str = "float32"
    use_cache: bool = True

    # attn
    attention_head: int = 8
    kv_head: int = 4
    head_dim: int = 32
    attention_bias: bool = False

    # rope
    base: float = 10_000
    original_max_position_embedding: int = 2048
    max_position_embedding: int = 2048
    rope_scaling: dict | None = None