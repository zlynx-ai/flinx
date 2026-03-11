# Training

Learn how to train any Zlynx model using the built-in `Trainer` — from data loading to checkpointing.

---

## Overview

The Zlynx training workflow has four components:

```
Dataset → DatasetConfig → Loss Function → TrainerConfig → Trainer.train()
```

1. **Dataset** — your raw data (list, HF dataset, or path)
2. **DatasetConfig** — how to shuffle, preprocess, and feed data
3. **Loss function** — `(model, batch) → scalar`
4. **TrainerConfig** — hyperparameters, logging, checkpointing

---

## Step 1: Prepare Your Dataset

Zlynx accepts several dataset formats:

| Format                       | Example                          |
| ---------------------------- | -------------------------------- |
| **List of dicts**            | `[{"x": array, "y": 1}, ...]`    |
| **HF `datasets.Dataset`**    | `datasets.load_dataset("mnist")` |
| **HF dataset name** (string) | `"openai/gsm8k"`                 |
| **Dict of arrays**           | `{"x": big_array, "y": labels}`  |

The simplest format is a list of dicts:

```python
train_data = [
    {"input": x_train[i], "target": y_train[i]}
    for i in range(len(x_train))
]
```

You can use any key names — just be consistent with your preprocessing and loss functions.

---

## Step 2: Configure the Dataset

```python
from zlynx.trainer import DatasetConfig

dsconfig = DatasetConfig(
    shuffle=True,
    shuffle_seed=42,
    preprocessing_fn=preprocess,   # optional
    filter_fn=None,                # optional
)
```

### The `preprocessing_fn`

This function is called on **each individual example** before batching. It receives a single dict and must return a dict:

```python
def preprocess(example):
    # example = {"input": array(28, 28), "target": 5}
    image = example["input"].astype(jnp.float32) / 255.0
    image = image[..., None]   # add channel dim
    return {"input": image, "target": example["target"]}
```

> [!IMPORTANT]
> `preprocessing_fn` is applied **per-example**, not per-batch. Zlynx uses Google Grain internally: it runs `.map(preprocessing_fn)` first, then `.batch()` to stack examples together.

### The `filter_fn`

An optional predicate to drop examples:

```python
def filter_fn(example):
    return example["target"] < 5   # only keep digits 0–4
```

### Loading HF Datasets Directly

Instead of preparing data yourself, pass a dataset name and configure in `DatasetConfig`:

```python
dsconfig = DatasetConfig(
    path="mnist",              # HF dataset name
    split="train",
    subset=None,               # optional config name
    preprocessing_fn=preprocess,
    shuffle=True,
)

# Then pass the path string as the dataset:
trainer = Trainer(model=model, dataset="mnist", ..., dsconfig=dsconfig)
```

---

## Step 3: Define the Loss Function

The loss function receives the **model** and a **batched** dict. It must return a scalar and be JIT-compatible:

```python
import optax

def loss_fn(model, batch):
    logits = model(batch["input"])       # forward pass
    labels = batch["target"]
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    return loss.mean()
```

> [!IMPORTANT]
> By the time data reaches `loss_fn`, Grain has already batched it. So `batch["input"]` has shape `(batch_size, ...)`.

### Returning Extra Metrics

Your loss function can return a **dict** instead of a scalar. The `"loss"` key is used for backpropagation, and everything else is forwarded to custom logging functions:

```python
def loss_fn(model, batch):
    logits = model(batch["input"])
    labels = batch["target"]
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

    preds = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(preds == labels)

    return {"loss": loss, "accuracy": accuracy}
```

Then use `logging_fn` in `TrainerConfig` to log those extra values:

```python
trconfig = TrainerConfig(
    ...,
    logging_fn={
        "accuracy": lambda **kw: kw["accuracy"],
    },
)
```

---

## Step 4: Configure the Trainer

```python
from zlynx.trainer import TrainerConfig

trconfig = TrainerConfig(
    # Batch
    batch_size=64,
    gradient_accumulation_steps=1,

    # Optimizer
    optimizer="adamw",
    learning_rate=1e-3,
    weight_decay=0.01,
    max_grad_norm=1.0,

    # Schedule
    lr_scheduler="warmup_cosine",
    warmup_steps=100,

    # Duration
    num_epochs=3,
    # max_steps=-1,               # set to override epochs

    # Checkpointing
    output_dir="./output",
    save_steps=500,
    save_total_limit=2,

    # Logging
    logging_steps=100,
    log_to=["stdout"],

    # Device
    sharding=False,                # single device (see sharding tutorial)
)
```

### Optimizers

| Value            | Optimizer                            |
| ---------------- | ------------------------------------ |
| `"adamw"`        | AdamW (default)                      |
| `"adam"`         | Adam                                 |
| `"sgd"`          | SGD                                  |
| `"lion"`         | Lion                                 |
| `"galore_adamw"` | AdamW + Gradient Low-Rank Projection |

### Learning Rate Schedules

| Value             | Behavior                     |
| ----------------- | ---------------------------- |
| `"constant"`      | Fixed learning rate          |
| `"linear"`        | Linear decay to 0            |
| `"cosine"`        | Cosine decay to 0 (default)  |
| `"warmup_cosine"` | Linear warmup → cosine decay |

### Gradient Accumulation

Simulate larger batch sizes on limited hardware:

```python
trconfig = TrainerConfig(
    batch_size=16,                       # micro-batch
    gradient_accumulation_steps=4,       # effective batch = 16 × 4 = 64
)
```

The Trainer handles accumulating gradients across micro-batches and averaging them before each optimizer step.

### Logging Backends

```python
trconfig = TrainerConfig(
    log_to=["stdout", "wandb", "tensorboard", "json"],
    run_name="my-experiment",
)
```

| Backend         | Output                                     |
| --------------- | ------------------------------------------ |
| `"stdout"`      | Prints to console                          |
| `"wandb"`       | Weights & Biases (auto-inits a run)        |
| `"tensorboard"` | TensorBoard logs in `output_dir/tb_logs/`  |
| `"json"`        | JSONL file at `output_dir/train_log.jsonl` |

---

## Step 5: Train

```python
from zlynx.trainer import Trainer

trainer = Trainer(
    model=model,
    dataset=train_data,
    loss_fn=loss_fn,
    trconfig=trconfig,
    dsconfig=dsconfig,
)

trainer.train()
```

The Trainer handles everything:

- Building the optimizer and LR schedule
- JIT-compiled training steps
- Gradient accumulation
- Periodic logging and checkpointing
- Orbax checkpoint rotation (keeps only `save_total_limit` most recent)

### Expected Output

```
step 100 | loss: 0.4521 | step: 100 | epoch: 0 | steps_per_sec: 8.12
step 200 | loss: 0.1203 | step: 200 | epoch: 0 | steps_per_sec: 9.34
...
saved checkpoint → step 500
...
training complete — 2814 steps | saved → ./output
```

---

## Quick Reference

```python
# Minimal training setup
from zlynx.trainer import Trainer, TrainerConfig, DatasetConfig

trainer = Trainer(
    model=model,
    dataset=data,                           # list, HF dataset, or string
    loss_fn=lambda model, batch: ...,       # (model, batch) → scalar
    trconfig=TrainerConfig(batch_size=32, num_epochs=5),
    dsconfig=DatasetConfig(shuffle=True),
)
trainer.train()
```

---

## Next Steps

Learn how to save and load your trained model in [Save & Load](./04_save_and_load.md).
