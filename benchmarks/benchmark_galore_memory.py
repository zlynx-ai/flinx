import jax
import jax.numpy as jnp
import optax
from flinx.trainer.optim import build_optimizer
from flinx.trainer.trainer import TrainerConfig

def compute_pytree_bytes(tree):
    leaves, _ = jax.tree_util.tree_flatten(tree)
    total_bytes = 0
    for leaf in leaves:
        if hasattr(leaf, 'nbytes'):
            total_bytes += leaf.nbytes
    return total_bytes

def run_memory_benchmark():
    print("=== GaLore vs. AdamW Memory Footprint Benchmark ===")
    
    # 1. Create a dummy large parameter module to simulate a Linear layer
    # Let's say a 4096 x 4096 projection layer (e.g. LLaMA 7B attention projection)
    m, n = 4096, 4096
    print(f"\nTarget Layer size: {m} x {n} float32 = {(m*n*4)/(1024**2):.2f} MB")
    
    params = {
        "kernel": jnp.zeros((m, n), dtype=jnp.float32),
        "bias": jnp.zeros((n,), dtype=jnp.float32)
    }
    
    # 2. Configure Standard AdamW
    adamw_config = TrainerConfig(
        optimizer="adamw",
        learning_rate=1e-4,
        weight_decay=0.01
    )
    adamw_opt = build_optimizer(adamw_config, total_steps=1000)
    adamw_state = adamw_opt.init(params)
    adamw_bytes = compute_pytree_bytes(adamw_state)
    
    print(f"\n[Standard AdamW]")
    print(f"Optimizer State Size: {adamw_bytes / (1024**2):.2f} MB")
    
    # 3. Configure GaLore AdamW
    galore_r = 128
    galore_config = TrainerConfig(
        optimizer="galore_adamw",
        learning_rate=1e-4,
        weight_decay=0.01,
        optimizer_kwargs={"galore_r": galore_r, "galore_update_proj_gap": 200}
    )
    galore_opt = build_optimizer(galore_config, total_steps=1000)
    galore_state = galore_opt.init(params)
    galore_bytes = compute_pytree_bytes(galore_state)
    
    print(f"\n[GaLore AdamW (Rank r={galore_r})]")
    print(f"Optimizer State Size: {galore_bytes / (1024**2):.2f} MB")
    
    # 4. Compare
    reduction = (1 - (galore_bytes / adamw_bytes)) * 100
    print(f"\n=== Result summary ===")
    print(f"GaLore reduced optimizer memory footprint by {reduction:.2f}%!")
    print(f"Saved: {(adamw_bytes - galore_bytes)/(1024**2):.2f} MB per 4096x4096 layer.")
    
if __name__ == "__main__":
    run_memory_benchmark()
