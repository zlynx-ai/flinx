


from flax import struct




@struct.dataclass
class VisionConfig:
    ...


@struct.dataclass
class AudioConfig:
    ...


@struct.dataclass
class LanguageConfig:
    architecture: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    act_fn: str
    num_hidden_layers: int
    norm_eps: float
    bias: bool
    dtype: str
    param_dtype: str
    use_cache: bool

    # attn
    attention_head: int
    kv_head: int
    head_dim: int
    attention_bias: bool

    # rope
    base: float
    original_max_position_embedding: int
    max_position_embedding: int
    rope_scaling: dict | None


@struct.dataclass
class ModelConfig:
    language_config: LanguageConfig | None = None
    vision_config: VisionConfig | None = None
    audio_config: AudioConfig | None = None


