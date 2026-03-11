from dataclasses import dataclass
from typing import Callable, Optional
import jax
import jax.numpy as jnp
from flax import nnx
from .trainer import Trainer, TrainerConfig

def token_log_probs(logits, labels):
    """Per-token log P(label | context).  (B, S-1)"""
    lp = jax.nn.log_softmax(logits[:, :-1, :], axis=-1)
    return jnp.take_along_axis(lp, labels[:, 1:, None], axis=-1).squeeze(-1)

def grpo_loss_fn(model, batch):
    """
    Standard Group Relative Policy Optimization loss.
    Expects `batch` to contain:
    - input_ids (B, S)
    - attention_mask (B, S)
    - advantages (B,)
    - old_lp (B, S-1)
    - ref_lp (B, S-1)
    - prompt_len (scalar int)
    - clip_eps (scalar float)
    - kl_coeff (scalar float)
    - entropy_coeff (scalar float)
    """
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    advantages = batch["advantages"]
    old_lp = batch["old_lp"]
    ref_lp = batch["ref_lp"]
    
    prompt_len = batch["prompt_len"]
    clip_eps = batch.get("clip_eps", 0.2)
    kl_coeff = batch.get("kl_coeff", 0.04)
    entropy_coeff = batch.get("entropy_coeff", 0.01)

    # Forward pass on current policy
    pos = jnp.maximum(jnp.cumsum(attention_mask.astype(jnp.int32), axis=-1) - 1, 0)
    outputs = model(input_ids, attention_mask, pos)
    logits = outputs.logits
    cur_lp = token_log_probs(logits, input_ids)

    # Completion mask (only evaluate loss on generated tokens)
    seq_len = cur_lp.shape[1]
    comp_mask = (jnp.arange(seq_len) >= (prompt_len - 1)) & (attention_mask[:, 1:] > 0)

    # 1. Policy gradient
    ratio = jnp.exp(cur_lp - old_lp)
    adv = advantages[:, None]
    pg = jnp.minimum(
        ratio * adv,
        jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv,
    )

    # 2. KL penalty
    log_r = ref_lp - cur_lp
    kl = jnp.exp(log_r) - log_r - 1

    # 3. Entropy bonus
    probs = jax.nn.softmax(logits[:, :-1, :], axis=-1)
    ent = -(probs * jax.nn.log_softmax(logits[:, :-1, :], axis=-1)).sum(-1)

    # Aggregate (TRL Default / DAPO formulation)
    # Instead of averaging the per-sequence averages (1/G * sum(1/|o_i| * sum(...))),
    # TRL divides the global sum of token losses by the global sum of valid tokens.
    total_tokens = comp_mask.sum().clip(min=1)
    
    pg_loss = -((pg * comp_mask).sum() / total_tokens)
    kl_loss = ((kl * comp_mask).sum() / total_tokens)
    ent_loss = -((ent * comp_mask).sum() / total_tokens) # Maximise entropy

    loss = pg_loss + kl_coeff * kl_loss + entropy_coeff * ent_loss
    return loss

@dataclass
class GRPOConfig(TrainerConfig):
    """Configuration for Group Relative Policy Optimization (GRPO)."""
    max_seq_len: int = 1024
    num_generations: int = 8               # G = Group size
    max_prompt_length: int = 512
    max_completion_length: int = 512       # Ensure prompt + compl = max_seq_len
    beta: float = 0.0                      # KL Divergence penalty
    clip_eps: float = 0.2                  # PPO clipping ratio
    entropy_coeff: float = 0.0             # Entropy bonus coefficient
    mu_epochs: int = 1                     # Inner optimization epochs per generation
    dataset_prompt_field: str = "prompt"
    dataset_responses_field: str = "responses"
    
class GRPOTrainer(Trainer):
    """
    Group Relative Policy Optimization Trainer.
    (Placeholder for GRPO-specific group sampling and reward formulation)
    """
    def __init__(
        self,
        model,
        ref_model, # Required for GRPO KL penalty
        reward_funcs: list[Callable], # E.g., rule-based metrics
        dataset,
        processor=None,
        trconfig: Optional[GRPOConfig] = None,
        dsconfig=None,
        loss_fn: Optional[Callable] = None,
    ):
        trconfig = trconfig or GRPOConfig()
        self.ref_model = ref_model
        self.reward_funcs = reward_funcs
        
        # Default to standard GRPO loss if none provided
        loss_fn = loss_fn or grpo_loss_fn
        
        super().__init__(
            model=model,
            dataset=dataset,
            loss_fn=loss_fn,
            processor=processor,
            trconfig=trconfig,
            dsconfig=dsconfig
        )

    def train(self):
        """
        Run the GRPO training loop with generation rollouts.
        For each batch (prompt):
        1. Generate G completions using the current policy (model.generate)
        2. Evaluate string completions through reward_funcs
        3. Compute group-relative advantages
        4. Capture static log probabilities (old_lp, ref_lp)
        5. Map a synthetic batch dictionary and run optimization steps
        """
        import time
        from pathlib import Path
        import orbax.checkpoint as ocp
        import numpy as np
        
        from .trainer import build_optimizer, Logger

        cfg = self.trconfig
        total_steps = cfg.max_steps
        if total_steps <= 0:
            raise ValueError("GRPOTrainer requires an explicit max_steps configuration.")

        # Optimizer
        opt = build_optimizer(cfg, total_steps)
        optimizer = nnx.Optimizer(self.model, opt, wrt=nnx.Param)

        # Checkpoints
        output_dir = Path(cfg.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        ckpt_mgr = ocp.CheckpointManager(
            output_dir / "checkpoints",
            options=ocp.CheckpointManagerOptions(max_to_keep=cfg.save_total_limit, create=True),
        )

        logger = Logger(cfg.log_to, cfg.output_dir, cfg.run_name)

        # Retrieve parameters
        G = cfg.num_generations
        max_prompt_len = cfg.max_prompt_length
        max_seq_len = cfg.max_seq_len
        max_new_tokens = max_seq_len - max_prompt_len

        global_step = 0
        t_start = time.time()
        
        from ..models.infer import _forward_step

        def _micro_forward_log_probs(mdl, ids, mask):
            """Forward function to get tokens log-probs without triggering large JIT overheads."""
            pos = jnp.maximum(jnp.cumsum(mask.astype(jnp.int32), axis=-1) - 1, 0)
            logits = mdl(ids, mask, pos).logits
            return token_log_probs(logits, ids)

        for epoch in range(cfg.num_epochs):
            for batch_prompts in self.dataset:
                # `batch_prompts` should dictate purely the prompt input ids and attention mask
                prompt_ids = batch_prompts["input_ids"] # (B, P)
                prompt_mask = batch_prompts["attention_mask"] # (B, P)
                
                # Assume batch size 1 for rollouts for simplicity of generation loop mappings
                prompt_ids = prompt_ids[0:1]
                prompt_mask = prompt_mask[0:1]
                prompt_len = int(prompt_mask.sum())

                # 1. Generate G Completions
                rep_ids = jnp.repeat(prompt_ids, G, axis=0)
                rep_mask = jnp.repeat(prompt_mask, G, axis=0)
                
                # Use model properties to perform live generation
                full_ids = self.model.generate(
                    rep_ids, rep_mask,
                    key=jax.random.key(global_step),
                    max_new_tokens=max_new_tokens,
                )
                
                # Turn off cache usage for forward loops
                for _, m in nnx.iter_modules(self.model):
                    if hasattr(m, "use_cache"):
                        m.use_cache = False

                # 2. Extract texts and compute reward
                completions = [
                    self.processor.decode(full_ids[j, prompt_len:], skip_special_tokens=True)
                    for j in range(G)
                ]
                
                rewards = np.zeros(G, dtype=np.float32)
                for i, comp in enumerate(completions):
                    # Sum all user defined metrics
                    r = sum(func(comp) for func in self.reward_funcs)
                    rewards[i] = r
                
                # 3. Advantages [(r - μ) / σ]
                adv_std = rewards.std() + 1e-8
                advantages = jnp.array((rewards - rewards.mean()) / adv_std)
                
                # 4. Freeze log probs for optimization
                full_mask = jnp.ones_like(full_ids, dtype=jnp.int32)
                
                # Current Policy
                old_lp = jax.lax.stop_gradient(_micro_forward_log_probs(self.model, full_ids, full_mask))
                
                # Reference Policy
                if self.ref_model is not None:
                    for _, m in nnx.iter_modules(self.ref_model):
                        if hasattr(m, "use_cache"):
                            m.use_cache = False
                    ref_lp = jax.lax.stop_gradient(_micro_forward_log_probs(self.ref_model, full_ids, full_mask))
                else:
                    ref_lp = old_lp

                # 5. Inner Optimization Loop (PPO style mu_epochs)
                synthetic_batch = {
                    "input_ids": full_ids,
                    "attention_mask": full_mask,
                    "advantages": advantages,
                    "old_lp": old_lp,
                    "ref_lp": ref_lp,
                    "prompt_len": prompt_len,
                    "clip_eps": cfg.clip_eps,
                    "kl_coeff": cfg.beta,
                    "entropy_coeff": cfg.entropy_coeff,
                }
                
                for _ in range(cfg.mu_epochs):
                    loss, grads = nnx.value_and_grad(self.loss_fn, argnums=nnx.DiffState(0, nnx.Param))(
                        self.model, synthetic_batch
                    )
                    optimizer.update(self.model, grads)

                global_step += 1

                # 6. Logging
                if global_step % cfg.logging_steps == 0:
                    metrics = {
                        "loss": float(loss),
                        "reward_mean": float(rewards.mean()),
                        "reward_std": float(rewards.std()),
                        "learning_rate": float(opt.schedule(global_step) if hasattr(opt, "schedule") else cfg.learning_rate),
                        "global_step": global_step,
                        "epoch": epoch,
                    }
                    if cfg.logging_fn is not None:
                        for k, v in cfg.logging_fn.items():
                            metrics[k] = float(v(**metrics))
                    logger.log(metrics, step=global_step)

                # 7. Checkpointing
                if global_step % cfg.save_steps == 0:
                    model_state = nnx.state(self.model)
                    ckpt_mgr.save(global_step, args=ocp.args.StandardSave(model_state))

                if global_step >= total_steps:
                    break
            if global_step >= total_steps:
                break

        # Final save
        model_state = nnx.state(self.model)
        ckpt_mgr.save(global_step, args=ocp.args.StandardSave(model_state), force=True)
        ckpt_mgr.wait_until_finished()
        logger.close()
