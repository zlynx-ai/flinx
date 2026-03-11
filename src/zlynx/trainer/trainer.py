
from dataclasses import dataclass, field
from pathlib import Path
import json
import time

import jax
import jax.numpy as jnp
import optax

from flax import nnx
from orbax import checkpoint as ocp
from typing import Callable
from .dataset import DatasetConfig, process_dataset
from .model import process_model
from .optim import build_optimizer


@dataclass
class TrainerConfig:
    # ── batch / accumulation ──
    batch_size: int = 8
    gradient_accumulation_steps: int = 1  # effective batch = batch_size × this

    # ── optimizer ──
    optimizer: str = "adamw"                     # "adamw", "sgd", "lion", "muon", ...
    optimizer_kwargs: dict = field(default_factory=dict)  # extra kwargs forwarded to optax
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    max_grad_norm: float | None = 1.0            # gradient clipping (None = disabled)

    # ── schedule ──
    lr_scheduler: str = "cosine"                 # "cosine", "linear", "constant", "warmup_cosine"
    warmup_steps: int = 0
    warmup_ratio: float = 0.0                    # alternative: fraction of total steps

    # ── training duration ──
    max_steps: int = -1                          # -1 = use num_epochs instead
    num_epochs: int = 1

    # ── precision ──
    dtype: str = "bfloat16"                      # param / compute dtype
    grad_dtype: str | None = None                # gradient dtype (None = same as dtype)

    # ── sharding / parallelism ──
    sharding: str | int | bool | None = "auto"     # "auto", None (skip), "fsdp", "tp", "dp", int (device ID), False (single)
    mesh_shape: tuple[int, ...] | None = None    # custom device mesh shape

    # ── checkpointing ──
    output_dir: str = "./output"
    save_steps: int = 500
    save_total_limit: int | None = 3             # max checkpoints to keep (None = unlimited)
    resume_from: str | None = None               # checkpoint path to resume from

    # ── logging ──
    logging_steps: int = 10
    log_to: list[str] = field(default_factory=lambda: ["stdout"])  # ["stdout", "wandb", "tensorboard", "json"]
    run_name: str | None = None
    logging_fn: dict[str, Callable] | None = None  # custom metrics: {"perplexity": lambda **kw: jnp.exp(kw["loss"])}

    # ── evaluation ──
    eval_steps: int | None = None                # None = no eval during training
    eval_batch_size: int | None = None           # None = same as batch_size


# ─────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────

class Logger:
    """Multi-backend logger supporting stdout, TensorBoard, W&B, and JSON."""

    def __init__(self, backends: list[str], output_dir: str, run_name: str | None = None):
        self.backends = backends
        self.output_dir = Path(output_dir)
        self._tb_writer = None
        self._json_path = None

        if "tensorboard" in backends:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = self.output_dir / "tb_logs"
            tb_dir.mkdir(parents=True, exist_ok=True)
            self._tb_writer = SummaryWriter(log_dir=str(tb_dir))

        if "wandb" in backends:
            import wandb
            if not wandb.run:
                wandb.init(project=run_name or "zlynx", name=run_name)

        if "json" in backends:
            self._json_path = self.output_dir / "train_log.jsonl"
            self._json_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, metrics: dict, step: int):
        """Log metrics to all active backends."""
        if "stdout" in self.backends:
            parts = [f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()]
            print(f"step {step} | " + " | ".join(parts))

        if "tensorboard" in self.backends and self._tb_writer:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._tb_writer.add_scalar(k, v, step)
            self._tb_writer.flush()

        if "wandb" in self.backends:
            import wandb
            wandb.log(metrics, step=step)

        if "json" in self.backends and self._json_path:
            record = {"step": step, **{k: float(v) if isinstance(v, (int, float, jnp.ndarray)) else v for k, v in metrics.items()}}
            with open(self._json_path, "a") as f:
                f.write(json.dumps(record) + "\n")

    def close(self):
        if self._tb_writer:
            self._tb_writer.close()
        if "wandb" in self.backends:
            import wandb
            if wandb.run:
                wandb.finish()


# ─────────────────────────────────────────────────────────────
# Training step (JIT-compiled)
# ─────────────────────────────────────────────────────────────

def compute_loss_and_grads(model, loss_fn, batch):
    """Compute loss, aux data, and gradients for a single micro-batch.

    Args:
        model: An nnx.Module to differentiate w.r.t.
        loss_fn: Callable (model, batch) → scalar loss OR dict with "loss" key.
            If a dict is returned, "loss" is used for backprop and the full
            dict is forwarded to logging_fn as **kwargs.
        batch: A single micro-batch (dict, array, etc.).

    Returns:
        Tuple of (loss, aux, grads) where:
            loss: scalar loss value
            aux: dict of all returned values (always contains "loss")
            grads: pytree matching the model's parameters
    """
    def wrapped(model, batch):
        result = loss_fn(model, batch)
        if isinstance(result, dict):
            return result["loss"], result
        return result, {"loss": result}

    (loss, aux), grads = nnx.value_and_grad(
        wrapped, argnums=nnx.DiffState(0, nnx.Param), has_aux=True
    )(model, batch)
    return loss, aux, grads


# ─────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────

class Trainer:
    def __init__(
        self,
        model,
        dataset,
        loss_fn: Callable,
        processor=None,
        trconfig: TrainerConfig | None = None,
        dsconfig: DatasetConfig | None = None,
    ):
        """Create a Trainer.

        Args:
            model: An nnx.Module to train.
            dataset: HF dataset name (str), datasets.Dataset, IterableDataset, list, or dict.
            loss_fn: Callable with signature (model, batch) → scalar loss.
                Must be JIT-compatible (no Python side effects).
            processor: Optional tokenizer/processor. Stored for convenience but
                not used by the Trainer directly — pass it into loss_fn if needed.
            trconfig: Training hyperparameters. Defaults to TrainerConfig().
            dsconfig: Dataset processing options. Defaults to DatasetConfig().
        """
        self.processor = processor
        self.loss_fn = loss_fn
        self.trconfig = trconfig or TrainerConfig()
        self.dsconfig = dsconfig or DatasetConfig()
        self.dataset, self._num_examples = process_dataset(
            dataset,
            dsconfig=self.dsconfig,
            trconfig=self.trconfig,
        )

        self.model = process_model(model, self.trconfig)

    def train(self):
        """Run the full training loop.

        Handles:
            - Optimizer and LR schedule construction from config.
            - Gradient accumulation over micro-batches.
            - Multi-backend logging (stdout, TensorBoard, W&B, JSON).
            - Custom metric functions via trconfig.logging_fn.
            - Orbax CheckpointManager with automatic rotation.
            - Early stop at max_steps if set.
        """
        cfg = self.trconfig

        # ── estimate total steps ──
        total_steps = cfg.max_steps
        if total_steps <= 0:
            if self._num_examples is None:
                raise ValueError("max_steps must be set when using a streaming dataset")
            steps_per_epoch = self._num_examples // (cfg.batch_size * cfg.gradient_accumulation_steps)
            total_steps = steps_per_epoch * cfg.num_epochs

        # ── build optimizer ──
        opt = build_optimizer(cfg, total_steps)
        optimizer = nnx.Optimizer(self.model, opt, wrt=nnx.Param)

        # ── checkpoint manager (with built-in rotation) ──
        output_dir = Path(cfg.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        ckpt_options = ocp.CheckpointManagerOptions(
            max_to_keep=cfg.save_total_limit,
            create=True,
        )
        ckpt_mgr = ocp.CheckpointManager(
            output_dir / "checkpoints",
            options=ckpt_options,
        )

        # ── logger ──
        logger = Logger(cfg.log_to, cfg.output_dir, cfg.run_name)

        # ── training loop ──
        global_step = 0
        micro_step = 0
        accum_loss = 0.0
        accum_grads = None
        log_loss = 0.0
        t_start = time.time()

        for epoch in range(cfg.num_epochs):
            for batch in self.dataset:
                # ── forward + backward (micro-batch) ──
                loss, aux, grads = compute_loss_and_grads(self.model, self.loss_fn, batch)
                micro_step += 1

                # ── accumulate gradients ──
                if accum_grads is None:
                    accum_grads = grads
                else:
                    accum_grads = jax.tree.map(jnp.add, accum_grads, grads)
                accum_loss += loss.item()

                # ── update when accumulation is complete ──
                if micro_step % cfg.gradient_accumulation_steps == 0:
                    # average gradients
                    if cfg.gradient_accumulation_steps > 1:
                        accum_grads = jax.tree.map(
                            lambda g: g / cfg.gradient_accumulation_steps, accum_grads
                        )

                    optimizer.update(self.model, accum_grads)
                    accum_grads = None
                    global_step += 1
                    avg_micro_loss = accum_loss / cfg.gradient_accumulation_steps
                    log_loss += avg_micro_loss
                    accum_loss = 0.0

                    # ── logging ──
                    if global_step % cfg.logging_steps == 0:
                        elapsed = time.time() - t_start
                        avg_loss = log_loss / cfg.logging_steps

                        # base metrics + anything from loss_fn's dict return
                        metrics = {
                            "loss": avg_loss,
                            "step": global_step,
                            "epoch": epoch,
                            "steps_per_sec": cfg.logging_steps / elapsed,
                        }

                        # custom metric functions — receive full aux dict
                        if cfg.logging_fn:
                            for name, fn in cfg.logging_fn.items():
                                metrics[name] = float(fn(**aux))

                        logger.log(metrics, step=global_step)
                        log_loss = 0.0
                        t_start = time.time()

                    # ── checkpointing (orbax handles rotation via max_to_keep) ──
                    if cfg.save_steps and global_step % cfg.save_steps == 0:
                        _, state = nnx.split(self.model)
                        ckpt_mgr.save(global_step, args=ocp.args.StandardSave(state))
                        ckpt_mgr.wait_until_finished()
                        print(f"saved checkpoint → step {global_step}")

                    # max steps reached
                    if cfg.max_steps > 0 and global_step >= cfg.max_steps:
                        break

            if cfg.max_steps > 0 and global_step >= cfg.max_steps:
                break

        # ── save final ──
        _, state = nnx.split(self.model)
        ckpt_mgr.save(global_step, args=ocp.args.StandardSave(state))
        ckpt_mgr.wait_until_finished()
        logger.log({"status": "complete", "total_steps": global_step}, step=global_step)
        logger.close()
        ckpt_mgr.close()
        print(f"training complete — {global_step} steps | saved → {output_dir}")