from dataclasses import dataclass
import jax
import jax.numpy as jnp
from typing import Callable, Optional
from ..trainer.trainer import Trainer, TrainerConfig
from ..trainer.grpo import token_log_probs  # Reuse standard log-prob calculation

@dataclass
class DPOConfig(TrainerConfig):
    """Configuration for Direct Preference Optimization (DPO)."""
    beta: float = 0.1
    max_prompt_length: int = 512
    max_seq_len: int = 1024
    dataset_prompt_field: str = "prompt"
    dataset_chosen_field: str = "chosen"
    dataset_rejected_field: str = "rejected"
    loss_type: str = "sigmoid" # "sigmoid", "hinge", "ipo", "kto_pair"
    label_smoothing: float = 0.0

def dpo_loss_fn(model, batch):
    """
    Direct Preference Optimization loss.
    Expects `batch` to contain:
    - chosen_input_ids (B, S)
    - chosen_attention_mask (B, S)
    - rejected_input_ids (B, S)
    - rejected_attention_mask (B, S)
    - prompt_len (scalar int)
    - ref_chosen_lp (B,) Total log prob of chosen completion under reference model
    - ref_rejected_lp (B,) Total log prob of rejected completion under reference model
    - beta (scalar float)
    - loss_type (string id)
    - label_smoothing (scalar float)
    """
    beta = batch.get("beta", 0.1)
    loss_type = batch.get("loss_type", 0) # 0: sigmoid, 1: hinge, 2: ipo, 3: kto_pair
    label_smoothing = batch.get("label_smoothing", 0.0)
    prompt_len = batch["prompt_len"]
    
    # 1. Forward Pass Chosen
    ch_ids = batch["chosen_input_ids"]
    ch_mask = batch["chosen_attention_mask"]
    ch_pos = jnp.maximum(jnp.cumsum(ch_mask.astype(jnp.int32), axis=-1) - 1, 0)
    
    ch_logits = model(ch_ids, ch_mask, ch_pos).logits
    ch_lp = token_log_probs(ch_logits, ch_ids)           # (B, S-1)
    
    # Ignore prompt tokens
    seq_len = ch_lp.shape[1]
    ch_comp_mask = (jnp.arange(seq_len) >= (prompt_len[:, None] - 1)) & (ch_mask[:, 1:] > 0)
    policy_chosen_logps = (ch_lp * ch_comp_mask).sum(-1) # (B,)
    
    # 2. Forward Pass Rejected
    rej_ids = batch["rejected_input_ids"]
    rej_mask = batch["rejected_attention_mask"]
    rej_pos = jnp.maximum(jnp.cumsum(rej_mask.astype(jnp.int32), axis=-1) - 1, 0)
    
    rej_logits = model(rej_ids, rej_mask, rej_pos).logits
    rej_lp = token_log_probs(rej_logits, rej_ids)        # (B, S-1)
    
    rej_comp_mask = (jnp.arange(seq_len) >= (prompt_len[:, None] - 1)) & (rej_mask[:, 1:] > 0)
    policy_rejected_logps = (rej_lp * rej_comp_mask).sum(-1) # (B,)
    
    # 3. Reference Model Log Probs (provided in batch to avoid dual-forward pass in loss_fn)
    ref_chosen_logps = batch["ref_chosen_lp"]
    ref_rejected_logps = batch["ref_rejected_lp"]
    
    # 4. Computed Implicit Rewards
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = pi_logratios - ref_logratios
    
    # 5. Determine the mathematical loss type
    # For JIT compatibility, we use entirely explicit boolean masking for string equivalence
    is_sigmoid = loss_type == 0
    is_hinge = loss_type == 1
    is_ipo = loss_type == 2
    is_kto = loss_type == 3
    
    # Sigmoid (Standard DPO)
    loss_sigmoid = -jax.nn.log_sigmoid(beta * logits) * (1 - label_smoothing) - jax.nn.log_sigmoid(-beta * logits) * label_smoothing
    
    # Hinge Loss
    loss_hinge = jax.nn.relu(1 - beta * logits)
    
    # IPO Loss
    loss_ipo = (logits - 1/(2 * beta)) ** 2
    
    # KTO Pair Loss
    chosen_KL = policy_chosen_logps - ref_chosen_logps
    rejected_KL = policy_rejected_logps - ref_rejected_logps
    
    chosen_logratios = policy_chosen_logps - ref_chosen_logps
    rejected_logratios = policy_rejected_logps - ref_rejected_logps
    
    loss_kto = 1 - jax.nn.sigmoid(beta * (chosen_logratios - rejected_KL)) + 1 - jax.nn.sigmoid(beta * (chosen_KL - rejected_logratios))
    
    # Combine (only 1 will be active due to JIT integer mapping)
    loss = (loss_sigmoid * is_sigmoid) + (loss_hinge * is_hinge) + (loss_ipo * is_ipo) + (loss_kto * is_kto)
    
    return loss.mean()

class DPOTrainer(Trainer):
    """
    Direct Preference Optimization Trainer.
    (Placeholder for DPO-specific dataset formatting and pairwise loss computation)
    """
    def __init__(
        self,
        model,
        ref_model, # Required for DPO KL divergence penalty
        dataset,
        processor=None,
        trconfig: Optional[DPOConfig] = None,
        dsconfig=None,
        loss_fn: Optional[Callable] = None,
    ):
        from ..trainer.dataset import DatasetConfig
        import numpy as np

        trconfig = trconfig or DPOConfig()
        self.ref_model = ref_model
        
        loss_fn = loss_fn or dpo_loss_fn
        
        # Determine strict integer mapping for JIT loss branching
        loss_types = {"sigmoid": 0, "hinge": 1, "ipo": 2, "kto_pair": 3}
        loss_type_id = loss_types.get(trconfig.loss_type.lower(), 0)

        # Build automated processor mapping if tokenizer is provided
        if processor is not None and dsconfig is None:
            def dpo_preprocess(example):
                """
                Map a single HF dictionary row containing 'prompt', 'chosen', 'rejected'
                into 'chosen_input_ids', 'rejected_input_ids', etc.
                """
                prompt = example[trconfig.dataset_prompt_field]
                chosen = example[trconfig.dataset_chosen_field]
                rejected = example[trconfig.dataset_rejected_field]
                
                chosen_text = prompt + chosen
                rejected_text = prompt + rejected
                
                chosen_enc = processor(
                    chosen_text,
                    padding="max_length",
                    truncation=True,
                    max_length=trconfig.max_seq_len,
                    return_tensors="np"
                )
                
                rejected_enc = processor(
                    rejected_text,
                    padding="max_length",
                    truncation=True,
                    max_length=trconfig.max_seq_len,
                    return_tensors="np"
                )
                
                # Processor returns shape (1, S) or (S,), we ensure 1D shape (S,)
                c_ids = chosen_enc["input_ids"].squeeze()
                c_mask = chosen_enc["attention_mask"].squeeze()
                r_ids = rejected_enc["input_ids"].squeeze()
                r_mask = rejected_enc["attention_mask"].squeeze()
                
                return {
                    "chosen_input_ids": c_ids,
                    "chosen_attention_mask": c_mask,
                    "rejected_input_ids": r_ids,
                    "rejected_attention_mask": r_mask,
                    "prompt_len": np.array(trconfig.max_prompt_length, dtype=np.int32),
                    "beta": np.array(trconfig.beta, dtype=np.float32),
                    "loss_type": np.array(loss_type_id, dtype=np.int32),
                    "label_smoothing": np.array(trconfig.label_smoothing, dtype=np.float32),
                }
                
            dsconfig = DatasetConfig(preprocessing_fn=dpo_preprocess)

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
        Run the DPO training loop.
        Overrides the standard Trainer.train() to inject `ref_chosen_lp` 
        and `ref_rejected_lp` from the frozen reference model into each batch natively, 
        saving the loss function from needing to hold its own stateful references.
        """
        import time
        from pathlib import Path
        import orbax.checkpoint as ocp
        from flax import nnx
        import jax
        import jax.numpy as jnp
        
        from ..trainer.trainer import build_optimizer, Logger, compute_loss_and_grads

        cfg = self.trconfig
        total_steps = cfg.max_steps
        if total_steps <= 0:
            if self._num_examples is None:
                raise ValueError("max_steps must be set when using a streaming dataset")
            steps_per_epoch = self._num_examples // (cfg.batch_size * cfg.gradient_accumulation_steps)
            total_steps = steps_per_epoch * cfg.num_epochs

        opt = build_optimizer(cfg, total_steps)
        optimizer = nnx.Optimizer(self.model, opt, wrt=nnx.Param)

        output_dir = Path(cfg.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        ckpt_mgr = ocp.CheckpointManager(
            output_dir / "checkpoints",
            options=ocp.CheckpointManagerOptions(max_to_keep=cfg.save_total_limit, create=True),
        )

        logger = Logger(cfg.log_to, cfg.output_dir, cfg.run_name)
        
        # Micro JIT forward function for the reference model to prevent memory duplication in training graph
        @jax.jit
        def _get_ref_logps(ref_model, ch_ids, ch_mask, rej_ids, rej_mask, prompt_len):
            # 1. Chosen
            ch_pos = jnp.maximum(jnp.cumsum(ch_mask.astype(jnp.int32), axis=-1) - 1, 0)
            ch_logits = ref_model(ch_ids, ch_mask, ch_pos).logits
            ch_lp = token_log_probs(ch_logits, ch_ids)
            ch_comp_mask = (jnp.arange(ch_lp.shape[1]) >= (prompt_len - 1)) & (ch_mask[:, 1:] > 0)
            ch_logps = (ch_lp * ch_comp_mask).sum(-1)
            
            # 2. Rejected
            rej_pos = jnp.maximum(jnp.cumsum(rej_mask.astype(jnp.int32), axis=-1) - 1, 0)
            rej_logits = ref_model(rej_ids, rej_mask, rej_pos).logits
            rej_lp = token_log_probs(rej_logits, rej_ids)
            rej_comp_mask = (jnp.arange(rej_lp.shape[1]) >= (prompt_len - 1)) & (rej_mask[:, 1:] > 0)
            rej_logps = (rej_lp * rej_comp_mask).sum(-1)
            
            return ch_logps, rej_logps

        global_step = 0
        micro_step = 0
        accum_loss = 0.0
        accum_grads = None
        log_loss = 0.0
        
        for epoch in range(cfg.num_epochs):
            for batch in self.dataset:
                
                # --- INJECT REFERENCE LOG PROBS ---
                if self.ref_model is not None:
                    # Turn off KV cache to avoid state mutations
                    for _, m in nnx.iter_modules(self.ref_model):
                        if hasattr(m, "use_cache"):
                            m.use_cache = False
                            
                    ch_lp, rej_lp = jax.lax.stop_gradient(_get_ref_logps(
                        self.ref_model,
                        batch["chosen_input_ids"], batch["chosen_attention_mask"],
                        batch["rejected_input_ids"], batch["rejected_attention_mask"],
                        batch["prompt_len"][0] # Assuming uniform prompt length for batch mapping
                    ))
                else:
                    # Fallback (KL == 0 constraint absent)
                    batch_size = batch["chosen_input_ids"].shape[0]
                    ch_lp = jnp.zeros((batch_size,), dtype=jnp.float32)
                    rej_lp = jnp.zeros((batch_size,), dtype=jnp.float32)
                    
                batch["ref_chosen_lp"] = ch_lp
                batch["ref_rejected_lp"] = rej_lp
                # ----------------------------------

                # Standard Micro-Batch Loop
                loss, aux, grads = compute_loss_and_grads(self.model, self.loss_fn, batch)
                micro_step += 1

                if accum_grads is None:
                    accum_grads = grads
                else:
                    accum_grads = jax.tree.map(lambda a, g: a + g, accum_grads, grads)
                
                accum_loss += loss
                log_loss += float(loss)

                # Gradient optimization step
                if micro_step == cfg.gradient_accumulation_steps:
                    accum_grads = jax.tree.map(lambda g: g / cfg.gradient_accumulation_steps, accum_grads)
                    optimizer.update(self.model, accum_grads)
                    global_step += 1
                    
                    if global_step % cfg.logging_steps == 0:
                        avg_loss = log_loss / (cfg.logging_steps * cfg.gradient_accumulation_steps)
                        metrics = {
                            "loss": avg_loss,
                            "learning_rate": float(opt.schedule(global_step) if hasattr(opt, "schedule") else cfg.learning_rate),
                            "global_step": global_step,
                            "epoch": epoch,
                        }
                        # Add any dict outputs from dpo_loss_fn (if aux is returning one)
                        if isinstance(aux, dict):
                            for k, v in aux.items():
                                if k != "loss":
                                    metrics[k] = float(v)
                                    
                        if cfg.logging_fn is not None:
                            for k, v in cfg.logging_fn.items():
                                metrics[k] = float(v(**metrics))
                                
                        logger.log(metrics, step=global_step)
                        log_loss = 0.0

                    if global_step % cfg.save_steps == 0:
                        model_state = nnx.state(self.model)
                        ckpt_mgr.save(global_step, args=ocp.args.StandardSave(model_state))

                    micro_step = 0
                    accum_loss = 0.0
                    accum_grads = None

                    if global_step >= total_steps:
                        break
            if global_step >= total_steps:
                break

        model_state = nnx.state(self.model)
        ckpt_mgr.save(global_step, args=ocp.args.StandardSave(model_state), force=True)
        ckpt_mgr.wait_until_finished()
        logger.close()
