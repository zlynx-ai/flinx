# Save & Load

How to persist your trained models and restore them later.

---

## Saving a Model

Every `Z` model has a built-in `save()` method:

```python
model.save("./my_model")
```

This creates a directory containing:

- **Orbax checkpoint files** — the serialized model weights
- **`config.json`** (if the model has a `config` attribute) — architecture metadata

```
my_model/
├── config.json          ← only if model has a config
├── _METADATA
└── ...                  ← Orbax weight files
```

---

## Loading a Model

Use the **class method** `load()` on your model class:

```python
model, processor = CNN.load(
    "./my_model",
    key=jax.random.key(0),
    num_classes=10,
)
```

### Why pass `key` and `num_classes` again?

Flax NNX needs to reconstruct the **model structure** (shapes, layers, etc.) before it can load the saved weights into it. The `load()` method:

1. Calls your `__init__` with the provided args to build an abstract model skeleton
2. Loads the Orbax checkpoint into that skeleton

The `key` can be **any key** — it's only used for structural initialization. The saved weights overwrite all parameters.

### The Return Tuple

`load()` always returns `(model, processor)`:

- `model` — the loaded model with restored weights
- `processor` — a tokenizer/processor if the model class defines one (e.g. for language models). `None` for most custom models.

```python
# For models without a processor, just ignore the second value:
model, _ = CNN.load("./my_model", key=jax.random.key(0), num_classes=10)
```

---

## Loading Trainer Checkpoints

When you use the `Trainer`, it saves checkpoints automatically using Orbax's `CheckpointManager`:

```
output/checkpoints/
├── 500/                    ← checkpoint at step 500
│   └── default/
│       └── ...
├── 1000/                   ← checkpoint at step 1000
│   └── default/
│       └── ...
└── 2814/                   ← final checkpoint
    └── default/
        └── ...
```

To load a Trainer checkpoint, point to the `default` subdirectory:

```python
model, _ = CNN.load(
    "./output/checkpoints/2814/default",
    key=jax.random.key(0),
    num_classes=10,
)
```

> [!NOTE]
> The Trainer auto-rotates checkpoints based on `save_total_limit` in `TrainerConfig`. If set to `2`, only the 2 most recent checkpoints are kept.

---

## Loading Config-Based Models

For models that use a config dataclass (like `LlamaLanguageModel`), loading is slightly different:

```python
from zlynx import Z

# If the checkpoint directory contains config.json, Z figures out the architecture:
model, processor = Z.load("./llama-checkpoint")
```

Or load with a specific class:

```python
from zlynx.models.llama import LlamaLanguageModel, LlamaConfig

config = LlamaConfig(vocab_size=32000, hidden_size=2048, ...)
model, processor = LlamaLanguageModel.load("./llama-checkpoint", config=config)
```

### With dtype casting

```python
# Load in bfloat16 for memory efficiency
model, processor = LlamaLanguageModel.load("./llama-checkpoint", dtype="bfloat16")
```

---

## Checkpoint Format Summary

| Method             | What it saves                                         | How to load                                |
| ------------------ | ----------------------------------------------------- | ------------------------------------------ |
| `model.save(path)` | Orbax weights + config.json                           | `Model.load(path, **init_args)`            |
| Trainer auto-save  | Orbax weights in `output/checkpoints/{step}/default/` | `Model.load(path/to/default, **init_args)` |

---

## Next Steps

Learn how to scale to multiple devices in [Sharding](./05_sharding.md).
