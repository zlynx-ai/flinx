from typing import NamedTuple, Any
import optax
import jax
import jax.numpy as jnp

from .trainer import TrainerConfig


OPTIMIZERS = {
    "adamw": optax.adamw,
    "adam": optax.adam,
    "sgd": optax.sgd,
    "lion": optax.lion,
}

SCHEDULERS = {
    "constant": lambda lr, total, warmup: optax.constant_schedule(lr),
    "linear": lambda lr, total, warmup: optax.linear_schedule(lr, 0.0, total),
    "cosine": lambda lr, total, warmup: optax.cosine_decay_schedule(lr, total),
    "warmup_cosine": lambda lr, total, warmup: optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=lr, warmup_steps=warmup, decay_steps=total
    ),
}


class GaloreState(NamedTuple):
    inner_state: Any
    projector: Any
    step: jnp.ndarray

def galore_wrapper(inner_opt: optax.GradientTransformation, r: int = 128, update_proj_gap: int = 200, scale: float = 1.0):
    """
    Wraps an Optax optimizer to apply Gradient Low-Rank Projection (GaLore).
    This reduces the memory footprint of optimizer states (like Adam's moments) 
    by projecting gradients into a low-rank subspace.
    """
    def init_fn(params):
        def _get_lr_shape(p):
            if p.ndim < 2 or min(p.shape) <= r:
                return p
            m, n = p.shape
            return jnp.zeros((m, r) if m < n else (r, n), dtype=p.dtype)
            
        low_rank_params = jax.tree_util.tree_map(_get_lr_shape, params)
        inner_state = inner_opt.init(low_rank_params)
        
        def _init_proj(p):
            if p.ndim < 2 or min(p.shape) <= r:
                return None
            m, n = p.shape
            if m < n:
                return jnp.zeros((n, r), dtype=p.dtype)
            else:
                return jnp.zeros((m, r), dtype=p.dtype)
            
        projector = jax.tree_util.tree_map(_init_proj, params)
        return GaloreState(inner_state=inner_state, projector=projector, step=jnp.array(0, dtype=jnp.int32))

    def update_fn(updates, state, params=None):
        step = state.step
        is_update_step = (step % update_proj_gap) == 0

        class Packed:
            def __init__(self, u, p):
                self.u = u; self.p = p

        def process_tensor(update, param, proj):
            if proj is None:
                return Packed(update, proj)
                
            m, n = param.shape
            is_right = (m < n)
            
            def compute_new_proj():
                # SVD on float32 for stability
                U, S, Vh = jnp.linalg.svd(update.astype(jnp.float32), full_matrices=False)
                if is_right:
                    # Vh top r rows are (r, n). Transpose is (n, r).
                    return Vh[:r, :].T.astype(update.dtype)
                else:
                    # U top r cols are (m, r).
                    return U[:, :r].astype(update.dtype)
                    
            new_proj = jax.lax.cond(
                is_update_step,
                compute_new_proj,
                lambda: proj
            )
            
            if is_right:
                # right projection: G @ P -> (m, r)
                low_rank_update = jnp.dot(update, new_proj)
            else:
                # left projection: Q^T @ G -> (r, n)
                low_rank_update = jnp.dot(new_proj.T, update)
                
            return Packed(low_rank_update, new_proj)

        packed_tree = jax.tree_util.tree_map(
            process_tensor, updates, params, state.projector,
            is_leaf=lambda x: x is None
        )
        
        low_rank_updates = jax.tree_util.tree_map(
            lambda x: x.u, packed_tree, 
            is_leaf=lambda x: isinstance(x, Packed) or x is None
        )
        new_projectors = jax.tree_util.tree_map(
            lambda x: x.p, packed_tree, 
            is_leaf=lambda x: isinstance(x, Packed) or x is None
        )
        
        def _get_lr_params(orig_p, proj):
            if proj is None or orig_p is None: return orig_p
            m, n = orig_p.shape
            is_right = (m < n)
            
            if is_right: 
                return jnp.dot(orig_p, proj)
            else:
                return jnp.dot(proj.T, orig_p)
                
        # Also need is_leaf for None projectors
        lr_params = jax.tree_util.tree_map(
            _get_lr_params, params, new_projectors, 
            is_leaf=lambda x: x is None
        ) if params is not None else None
        
        inner_updates_lr, new_inner_state = inner_opt.update(low_rank_updates, state.inner_state, lr_params)
        
        def _project_up(inner_u, orig_p, proj):
            if proj is None: return inner_u * scale
            m, n = orig_p.shape
            is_right = (m < n)
            
            if is_right:
                # is_right -> inner_u was (m, r). proj is (n, r).
                # inner_u @ proj^T -> (m, n)
                full_u = jnp.dot(inner_u, proj.T)
            else:
                # left proj -> inner_u was (r, n). proj is (m, r).
                # proj @ inner_u -> (m, n)
                full_u = jnp.dot(proj, inner_u)
                
            return full_u * scale
            
        final_updates = jax.tree_util.tree_map(
            _project_up, inner_updates_lr, params, new_projectors,
            is_leaf=lambda x: x is None
        )
        
        return final_updates, GaloreState(inner_state=new_inner_state, projector=new_projectors, step=step + 1)

    return optax.GradientTransformation(init_fn, update_fn)


def build_optimizer(trconfig: "TrainerConfig", total_steps: int):
    """Build an optax optimizer chain from TrainerConfig."""
    warmup = trconfig.warmup_steps or int(trconfig.warmup_ratio * total_steps)

    schedule_fn = SCHEDULERS.get(trconfig.lr_scheduler, SCHEDULERS["cosine"])
    schedule = schedule_fn(trconfig.learning_rate, total_steps, warmup)

    opt_fn = OPTIMIZERS.get(trconfig.optimizer.replace("galore_", ""))
    if opt_fn is None:
        raise ValueError(f"Unknown optimizer: {trconfig.optimizer}. Available inner: {list(OPTIMIZERS.keys())}")

    inner_kwargs = {k: v for k, v in trconfig.optimizer_kwargs.items() if not k.startswith("galore_")}

    if "adamw" in trconfig.optimizer:
        opt = opt_fn(learning_rate=schedule, weight_decay=trconfig.weight_decay, **inner_kwargs)
    elif "sgd" in trconfig.optimizer:
        opt = opt_fn(learning_rate=schedule, **inner_kwargs)
    else:
        opt = opt_fn(learning_rate=schedule, **inner_kwargs)
        
    if trconfig.optimizer.startswith("galore_"):
        r = trconfig.optimizer_kwargs.get("galore_r", 128)
        update_gap = trconfig.optimizer_kwargs.get("galore_update_proj_gap", 200)
        scale = trconfig.optimizer_kwargs.get("galore_scale", 1.0)
        opt = galore_wrapper(opt, r=r, update_proj_gap=update_gap, scale=scale)

    if trconfig.max_grad_norm is not None:
        opt = optax.chain(optax.clip_by_global_norm(trconfig.max_grad_norm), opt)

    return opt