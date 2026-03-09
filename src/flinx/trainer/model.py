

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


def count_params(model) -> int:
    """Count total number of parameters in a model."""
    _, state = nnx.split(model)
    leaves = jax.tree.leaves(state)
    return sum(p.size for p in leaves if hasattr(p, 'size'))


def param_bytes(model, dtype=jnp.float32) -> int:
    """Estimate model memory in bytes for a given dtype."""
    return count_params(model) * jnp.dtype(dtype).itemsize


def process_model(model, trconfig):
    """Apply sharding/placement to a model based on TrainerConfig.

    Args:
        model: An nnx.Module to shard or place.
        trconfig: TrainerConfig with sharding settings.
            sharding="auto": DDP if model fits on one device with >=1.5GB headroom, FSDP otherwise.
            sharding=None: do not change sharding (assumes user applied custom routing/sharding).
            sharding=False: place on first device only (single-device).
            sharding=<int>: place on device with given ID.
            sharding="dp": replicate across all devices (data parallelism).
            sharding="fsdp": shard params across all devices.
            sharding="tp": tensor parallelism.

    Returns:
        The model, placed/sharded according to config.
    """
    devices = jax.devices()
    sharding = trconfig.sharding

    # ── custom sharding (skip) ──
    if sharding is None:
        return model

    # ── explicit device ID ──
    if isinstance(sharding, int):
        if sharding >= len(devices):
            raise ValueError(
                f"Device ID {sharding} out of range. "
                f"Available devices: {len(devices)} (IDs 0-{len(devices) - 1})"
            )
        target = jax.devices()[sharding]
        state = nnx.state(model)
        state = jax.device_put(state, target)
        nnx.update(model, state)
        return model

    # ── single-device (no sharding) ──
    if sharding is False:
        target = devices[0]
        state = nnx.state(model)
        state = jax.device_put(state, target)
        nnx.update(model, state)
        return model

    # ── auto: decide based on memory ──
    if sharding == "auto":
        model_bytes = param_bytes(model)
        # check if model fits on a single device with headroom
        try:
            device_mem = devices[0].memory_stats()
            available = device_mem.get("bytes_limit", float("inf"))
        except Exception:
            available = float("inf")

        headroom = 1.5 * (1024 ** 3)  # 1.5 GiB
        if model_bytes + headroom < available and len(devices) > 1:
            sharding = "dp"
        elif len(devices) > 1:
            sharding = "fsdp"
        else:
            # single device, nothing to do
            return model

    # ── data parallelism: replicate across all devices ──
    if sharding == "dp":
        mesh = jax.sharding.Mesh(np.array(devices), ("dp",))
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        state = nnx.state(model)
        state = jax.device_put(state, replicated)
        nnx.update(model, state)
        return model

    # ── FSDP: shard params across devices ──
    if sharding == "fsdp":
        mesh = jax.sharding.Mesh(np.array(devices), ("fsdp",))
        sharded = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("fsdp"))
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

        state = nnx.state(model)

        def shard_param(p):
            if not hasattr(p, 'ndim'):
                return p
            if p.ndim >= 2:
                return jax.device_put(p, sharded)
            else:
                return jax.device_put(p, replicated)

        state = jax.tree.map(shard_param, state)
        nnx.update(model, state)
        return model

    # ── tensor parallelism ──
    if sharding == "tp":
        mesh = jax.sharding.Mesh(np.array(devices), ("tp",))
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

        state = nnx.state(model)

        def shard_tp_param(path, p):
            if not hasattr(p, "ndim"):
                return p

            # path is a tuple of nnx tree keys (DictKey, SequenceKey, etc.)
            path_str = ".".join(str(getattr(k, "key", getattr(k, "idx", k))) for k in path)

            pspec = None
            if "embed_tokens" in path_str and "embedding" in path_str:
                pspec = jax.sharding.PartitionSpec("tp", None)

            elif "lm_head" in path_str and "kernel" in path_str:
                pspec = jax.sharding.PartitionSpec(None, "tp")

            # MLP
            elif "gate_proj" in path_str and ("kernel" in path_str or "bias" in path_str):
                pspec = jax.sharding.PartitionSpec(None, "tp") if "kernel" in path_str else jax.sharding.PartitionSpec("tp")
            elif "up_proj" in path_str and ("kernel" in path_str or "bias" in path_str):
                pspec = jax.sharding.PartitionSpec(None, "tp") if "kernel" in path_str else jax.sharding.PartitionSpec("tp")
            elif "down_proj" in path_str and "kernel" in path_str:
                pspec = jax.sharding.PartitionSpec("tp", None)

            # Attention
            elif "q_proj" in path_str and ("kernel" in path_str or "bias" in path_str):
                pspec = jax.sharding.PartitionSpec(None, "tp") if "kernel" in path_str else jax.sharding.PartitionSpec("tp")
            elif "k_proj" in path_str and ("kernel" in path_str or "bias" in path_str):
                pspec = jax.sharding.PartitionSpec(None, "tp") if "kernel" in path_str else jax.sharding.PartitionSpec("tp")
            elif "v_proj" in path_str and ("kernel" in path_str or "bias" in path_str):
                pspec = jax.sharding.PartitionSpec(None, "tp") if "kernel" in path_str else jax.sharding.PartitionSpec("tp")
            elif "o_proj" in path_str and "kernel" in path_str:
                pspec = jax.sharding.PartitionSpec("tp", None)

            if pspec is not None:
                return jax.device_put(p, jax.sharding.NamedSharding(mesh, pspec))
            else:
                return jax.device_put(p, replicated)

        state = jax.tree_util.tree_map_with_path(shard_tp_param, state)
        nnx.update(model, state)
        return model

    raise ValueError(f"Unknown sharding strategy: {sharding}")