# 🐆 Flinx

**JAX/Flax library for easy model implementation and training**

---

Flinx is a lightweight, modular library built on top of [JAX](https://github.com/jax-ml/jax) and [Flax NNX](https://github.com/google/flax) for implementing and training deep learning models. It provides reusable building blocks and ready-to-use model architectures — all with first-class support for JAX's JIT compilation, automatic sharding, and checkpoint management.

## Installation

Requires **Python ≥ 3.12**.

```bash
# with uv (recommended)
uv pip install -e .

# or with pip
pip install -e .
```

### Dependencies

| Package               | Purpose                             |
| --------------------- | ----------------------------------- |
| `jax ≥ 0.9.1`         | XLA-accelerated numerical computing |
| `flax ≥ 0.12.5`       | Neural network library (NNX API)    |
| `optax ≥ 0.2.6`       | Gradient-based optimizers           |
| `orbax ≥ 0.1.9`       | Checkpoint management               |
| `tokenizers ≥ 0.22.2` | Fast tokenizer (HuggingFace)        |

## Quick Start

### Define a model from config

```python
from flinx.models.llama import LlamaConfig, LlamaLanguageModel

config = LlamaConfig(
    vocab_size=32000,
    hidden_size=2048,
    intermediate_size=5632,
    num_hidden_layers=22,
    attention_head=32,
    kv_head=4,
    head_dim=64,
)

model = LlamaLanguageModel(config)
```

### Load a pretrained checkpoint

```python
from flinx import Flinx

# auto-detects architecture from config.json
model, tokenizer = Flinx.load("./my-llama-ckpt", dtype="bfloat16")
```

### Generate text

```python
import jax

input_ids = tokenizer.encode("Once upon a time").ids
input_ids = jax.numpy.array([input_ids])  # add batch dim

output_ids = model.generate(
    input_ids,
    max_new_tokens=128,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.1,
    key=jax.random.key(42),
)

print(tokenizer.decode(output_ids[0].tolist()))
```

### Save a checkpoint

```python
model.save("./my-llama-ckpt")
```

## Features

- **Modular building blocks** — Attention (GQA/MQA), gated MLP, RMSNorm, RoPE, KV Cache — mix and match to define new architectures.
- **Multiple RoPE variants** — Standard, LLaMA 3, with stubs for linear, dynamic, YaRN, LongRoPE, and Hierarchical RoPE.
- **KV cache with auto-management** — Prefill + decode loop, cache initialization/update handled automatically during generation.
- **JIT-compiled generation** — `sample_token` and forward steps are `jax.jit`-compiled for hardware-accelerated autoregressive decoding.
- **Sampling strategies** — Temperature scaling, Top-K, Nucleus (Top-P), repetition penalty, token suppression, and EOS handling.
- **Checkpoint save/load via Orbax** — Sharding-aware restore across JAX devices with optional dtype casting on load.
- **Architecture auto-detection** — `Flinx.load()` reads `config.json` and instantiates the correct model class automatically.
- **HuggingFace-compatible tokenizer** — Full wrapper around `tokenizers` with encode, decode, batch support, and `tokenizer.json` loading.
- **Multimodal config stubs** — `LanguageConfig`, `VisionConfig`, `AudioConfig` ready for future multimodal architectures.

## Architecture

```
src/flinx/
├── models/
│   ├── base.py          # Flinx — base class with save/load
│   ├── config.py         # LanguageConfig, VisionConfig, AudioConfig, ModelConfig
│   ├── infer.py          # LanguageModel — generate(), sample_token()
│   ├── utils.py          # dtype / activation function helpers
│   ├── llama/            # LLaMA architecture (complete)
│   ├── deepseek/         # DeepSeek (WIP)
│   ├── gemma/            # Gemma (WIP)
│   ├── gptoss/           # GPToss (WIP)
│   └── siglip/           # SigLIP vision encoder (WIP)
├── modules/
│   ├── attn.py           # Multi-Head / Grouped-Query / Multi-Query Attention
│   ├── mlp.py            # Gated MLP (gate + up + down projections)
│   ├── norm.py           # RMSNorm
│   ├── rope.py           # Rotary Position Embedding (multiple variants)
│   ├── cache.py          # KV Cache base class
│   ├── embed.py          # Embedding utilities
│   └── moe.py            # Mixture of Experts (WIP)
├── processor/
│   └── tokenizer.py      # Tokenizer wrapper (HuggingFace tokenizers)
└── trainer/
    ├── sft.py            # Supervised fine-tuning
    └── optim.py          # Optimizer utilities
```

### How a model is composed

Each model architecture (e.g. LLaMA) is built by composing modules from `flinx.modules`:

```
LlamaLanguageModel (LanguageModel + Flinx)
 └── Llama
      ├── Embed
      ├── RotaryEmbedding
      ├── LlamaTransformer[] × N layers
      │    ├── RMSNorm (input)
      │    ├── Attention (GQA) + KVCache
      │    ├── RMSNorm (post-attention)
      │    └── MLP (gated)
      └── RMSNorm (final)
 └── lm_head (Linear)
```

To add a new architecture, extend `Flinx` + `LanguageModel` and wire together the modules you need:

```python
class MyLanguageModel(LanguageModel, Flinx):
    def __init__(self, config):
        nnx.Module.__init__(self)
        LanguageModel.__init__(self, config=config)
        # compose your modules here ...
```

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
