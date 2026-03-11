# Zlynx: Deep Dive Tutorial

Welcome to the Zlynx advanced features tutorial. This guide covers how to leverage the powerful optimizations we've built, including Automatic Sharding, the enhanced Trainer, and the vast suite of Parameter-Efficient Fine-Tuning (PEFT) techniques.

## 1. Automatic Sharding & Distributed Loading

Zlynx takes the headache out of large model partitioning. When loading a model, use the `sharding` argument. Zlynx uses `jax.sharding.NamedSharding` internally.

```python
from zlynx.models.base import Zlynx

# Options for 'sharding':
# - "fsdp": Fully Sharded Data Parallel (slices parameters and optimizer states across axis "fsdp")
# - "dp": Data Parallel (replicates model, shards data batches)
# - "tp": Tensor Parallel (slices specific weight matrices)
# - "auto": Fallback heuristic depending on model param count and device memory

model, processor = Zlynx.from_pretrained(
    "path/to/checkpoint",
    sharding="fsdp",
    dtype="bfloat16"
)
```

## 2. Advanced PEFT Architectures

Zlynx implements adapters straight into the `flax.nnx` module graph using an elegant tree-traversal replacement utility `apply_peft`.

### Applying Adapters

```python
from zlynx.trainer.peft import apply_peft

# In-place modifies the model by wrapping specified target modules
# with your chosen adapter.
model = apply_peft(
    model,
    method="lora", # Try: "dora", "vera", "loha", "lokr", "adalora"
    r=16,
    alpha=32,
    target_modules=["q_proj", "v_proj"]
)
```

### Overview of PEFT Methods Available:

- **LoRA**: Standard low-rank update `A @ B`.
- **DoRA**: Decomposes the weight into a learnable magnitude vector and a direction matrix updated via LoRA. Mimics full fine-tuning very closely.
- **VeRA**: Phenomenal parameter reduction. Randomly initializes and _freezes_ `A` and `B` matrices, only training tiny scaling vectors `d` and `b`.
- **LoHa**: Uses the Hadamard (element-wise) product of two separate low-rank paths constraint: `(A1 @ B1) * (A2 @ B2)`. Extremely high expressiveness.
- **LoKr**: Uses the Kronecker product of a learnable matrix and a frozen random matrix.
- **AdaLoRA**: Similar to LoRA but explicitly models singular values `P @ diag(E) @ Q`. Preemptively designed for dynamic rank pruning during training.

## 3. The GaLore Optimizer

GaLore (Gradient Low-Rank Projection) is an entirely different beast. It operates within the optimizer state, slashing the memory required for AdamW moments.

Just pass `"galore_adamw"` to the `TrainerConfig`:

```python
from zlynx.trainer.trainer import TrainerConfig

config = TrainerConfig(
    optimizer="galore_adamw",
    learning_rate=2e-5,
    optimizer_kwargs={
        "galore_r": 128,              # The rank of the SVD projection
        "galore_update_proj_gap": 200 # How often to recompute the SVD projectors
    }
)
```

## 4. Trainer Enhancements: Accumulation, Logging & Checkpointing

The Zlynx `Trainer` integrates tightly with major workflow elements:

- **Gradient Accumulation:** Use `gradient_accumulation_steps=N` inside `TrainerConfig` to simulate huge batches on limited VRAM hardware. Zlynx handles the inner micro-batch accumulation.
- **Orbax Checkpoints:** Safely auto-rotates weights using Google's `ocp.CheckpointManager` without arbitrary file I/O overhead.
- **Integrated Logging:** Pass `log_to=["stdout", "wandb", "json"]` completely natively. You can also inject `logging_fn` dicts to track custom perplexities or metrics per step!
