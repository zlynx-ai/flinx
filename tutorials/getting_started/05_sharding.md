# Sharding & Multi-Device Training

Scale your training across multiple GPUs or TPUs with a single config change.

---

## Overview

Zlynx handles device placement and model partitioning through the `sharding` option in `TrainerConfig`. Under the hood, it uses JAX's `NamedSharding` and `Mesh` APIs — but you don't need to touch those directly.

```python
trconfig = TrainerConfig(
    sharding="auto",   # ← this is all you need
    ...
)
```

---

## Sharding Strategies

| Value    | Strategy                                                      | Best for                                        |
| -------- | ------------------------------------------------------------- | ----------------------------------------------- |
| `"auto"` | Auto-detect based on model size and device memory             | Most cases — let Zlynx decide                   |
| `"dp"`   | Data Parallel — replicate model, split data                   | Model fits on one device                        |
| `"fsdp"` | Fully Sharded Data Parallel — shard parameters across devices | Large models that don't fit on one device       |
| `"tp"`   | Tensor Parallel — shard specific weight matrices              | Very large models with transformer architecture |
| `False`  | Single device (no distribution)                               | Debugging, single GPU                           |
| `None`   | Skip — assume user applied custom sharding                    | Advanced use cases                              |
| `<int>`  | Place on a specific device by ID                              | Targeting a particular GPU                      |

---

## Auto Sharding

The default `"auto"` strategy makes a smart choice:

1. **If the model fits on one device** (with ≥1.5 GB headroom) **and** there are multiple devices → uses **Data Parallel** (`"dp"`)
2. **If the model is too large** for one device → uses **FSDP**
3. **Single device** → does nothing (no sharding needed)

```python
trconfig = TrainerConfig(
    sharding="auto",
    batch_size=64,
    ...
)
```

This is the recommended default for most workloads.

---

## Data Parallel (DP)

Replicates the full model on every device. Each device processes a different slice of the batch. Gradients are averaged across devices.

```python
trconfig = TrainerConfig(
    sharding="dp",
    batch_size=64,   # total batch across all devices
    ...
)
```

**When to use:** Your model fits comfortably on a single device and you want faster training through data parallelism.

---

## Fully Sharded Data Parallel (FSDP)

Shards model parameters across devices. Each device holds only a fraction of the weights, dramatically reducing per-device memory.

```python
trconfig = TrainerConfig(
    sharding="fsdp",
    batch_size=64,
    ...
)
```

Zlynx automatically decides how to shard:

- **2D+ tensors** (weight matrices) → sharded across the `"fsdp"` mesh axis
- **1D tensors** (biases, norms) → replicated (too small to shard)

**When to use:** Large models that don't fit on a single device's memory.

---

## Tensor Parallel (TP)

Splits individual weight matrices across devices. Zlynx applies sensible partitioning rules for transformer architectures:

```python
trconfig = TrainerConfig(
    sharding="tp",
    batch_size=64,
    ...
)
```

The default TP partitioning:

| Layer                                | Partitioning                 |
| ------------------------------------ | ---------------------------- |
| `embed_tokens.embedding`             | Shard rows across TP axis    |
| `lm_head.kernel`                     | Shard columns across TP axis |
| `q_proj`, `k_proj`, `v_proj` kernels | Shard columns                |
| `o_proj.kernel`                      | Shard rows                   |
| `gate_proj`, `up_proj` kernels       | Shard columns                |
| `down_proj.kernel`                   | Shard rows                   |
| Everything else                      | Replicated                   |

**When to use:** Very large transformer models where you need intra-layer parallelism.

---

## Single Device

Force everything onto one device:

```python
# Use the first device
trconfig = TrainerConfig(sharding=False, ...)

# Or pick a specific device by index
trconfig = TrainerConfig(sharding=2, ...)   # device ID 2
```

---

## Custom Sharding

If you want full control, set `sharding=None` and apply your own JAX sharding before creating the Trainer:

```python
import jax
import numpy as np
from flax import nnx

# Create your own mesh
devices = jax.devices()
mesh = jax.sharding.Mesh(np.array(devices), ("data",))
replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

# Apply to model
state = nnx.state(model)
state = jax.device_put(state, replicated)
nnx.update(model, state)

# Tell Trainer to skip sharding
trconfig = TrainerConfig(sharding=None, ...)
```

---

## Checking Your Devices

Before training, verify what JAX sees:

```python
import jax

print(f"Backend:  {jax.default_backend()}")
print(f"Devices:  {jax.devices()}")
print(f"Count:    {len(jax.devices())}")
```

### Simulating Multiple Devices (for testing)

On a single machine, you can simulate multiple CPU devices:

```python
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import jax
print(jax.devices())   # [CpuDevice(id=0), CpuDevice(id=1), ...]
```

> [!IMPORTANT]
> Set `XLA_FLAGS` **before** importing JAX. It won't work if JAX is already initialized.

---

## Example: Multi-GPU Training

```python
import jax
from zlynx.trainer import Trainer, TrainerConfig, DatasetConfig

print(f"Training on {len(jax.devices())} {jax.default_backend().upper()} devices")

trconfig = TrainerConfig(
    batch_size=128,
    learning_rate=1e-3,
    num_epochs=5,
    sharding="auto",                 # auto-selects DP or FSDP
    output_dir="./multi_gpu_output",
    log_to=["stdout", "wandb"],
)

trainer = Trainer(
    model=model,
    dataset=train_data,
    loss_fn=loss_fn,
    trconfig=trconfig,
    dsconfig=DatasetConfig(shuffle=True, preprocessing_fn=preprocess),
)

trainer.train()
```

---

## Strategy Decision Flowchart

```
Does your model fit on one device?
├── YES → Use "dp" (or "auto")
└── NO
    ├── Is it a transformer? → Try "tp"
    └── Otherwise → Use "fsdp"

Not sure? → Use "auto" and let Zlynx decide.
```

---

## Next Steps

You've completed the Getting Started series! Here's where to go next:

- **[MNIST Tutorial](../mnist.md)** — full end-to-end example
- **[PEFT Adapters](../advanced/01_peft.md)** — LoRA, DoRA, VeRA, and more
- **[GaLore Optimizer](../advanced/02_galore.md)** — memory-efficient full-parameter training
