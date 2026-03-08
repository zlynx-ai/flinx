




from flax import nnx
import functools
import jax, jax.numpy as jnp



@functools.partial(
    jax.jit, static_argnames=("temperature", "top_k", "top_p", "repetition_penalty")
)
def sample_token(
    logits, input_ids, input_mask, key, temperature=1.0, top_k=50, top_p=1.0, repetition_penalty=1.0
):
    if temperature == 0.0:
        return jax.lax.argmax(logits, axis=1, index_dtype=jnp.int32), key

    logits = logits / temperature

    # Repetition Penalty
    if repetition_penalty != 1.0:
        one_hots = jax.nn.one_hot(input_ids, logits.shape[-1])
        valid_one_hots = jnp.where(jnp.expand_dims(input_mask, -1), one_hots, 0.0)
        score_mask = valid_one_hots.any(axis=1)
        penalized_logits = jnp.where(
            logits > 0, logits / repetition_penalty, logits * repetition_penalty
        )
        logits = jnp.where(score_mask, penalized_logits, logits)

    # Top-K Sampling
    if top_k > 0:
        top_k_vals, _ = jax.lax.top_k(logits, top_k)
        min_vals = top_k_vals[:, -1:]
        logits = jnp.where(logits < min_vals, -jnp.inf, logits)

    # Nucleus (Top-P) Sampling
    if top_p < 1.0:
        sorted_indices = jnp.argsort(logits, axis=-1)[:, ::-1]
        sorted_logits = jnp.take_along_axis(logits, sorted_indices, axis=-1)

        cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=-1), axis=-1)

        mask = cumulative_probs > top_p
        mask = jnp.roll(mask, 1, axis=-1)
        mask = mask.at[:, 0].set(False)  # Always keep at least the most probable token

        inv_indices = jnp.argsort(sorted_indices, axis=-1)
        mask_in_original_order = jnp.take_along_axis(mask, inv_indices, axis=-1)

        logits = jnp.where(mask_in_original_order, -jnp.inf, logits)

    key, subkey = jax.random.split(key)
    result = jax.random.categorical(subkey, logits, axis=-1).astype(jnp.int32)
    return result, key


@nnx.jit
def _forward_step(model, input_ids, attention_mask, position_ids):
    return model(input_ids, attention_mask, position_ids)


class LanguageModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.config = kwargs.get("config", None)

    def set_config(self, **kwargs):
        if self.config is not None:
            if hasattr(self.config, "replace"):
                self.config = self.config.replace(**kwargs)
            else:
                for k, v in kwargs.items():
                    setattr(self.config, k, v)
            self.kwargs["config"] = self.config

    def init_cache(self, batch_size: int, max_seq_len: int):
        from ..modules.cache import KVCacheBase

        for _, module in nnx.iter_modules(self):
            if isinstance(module, KVCacheBase):
                module.init_cache_state(batch_size, max_seq_len)

    def generate(
        self,
        input_ids: jax.Array,
        attention_mask: jax.Array = None,
        key: jax.Array | None = None,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        eos_token_id: int | None = None,
        suppress_tokens: list[int] | None = None,
    ):
        B, S = input_ids.shape
        max_len = S + max_new_tokens

        cfg = getattr(self, "config", getattr(self, "kwargs", {}).get("config", None))
        use_cache = True if cfg is None else getattr(cfg, "use_cache", True)

        # Enforce use_cache on all submodules (like Attention) since checkpoint config
        # might have use_cache=False and model.config.replace() doesn't update instances
        from ..modules.cache import KVCacheBase

        for _, module in nnx.iter_modules(self):
            if isinstance(module, KVCacheBase) and hasattr(module, "use_cache"):
                module.use_cache = use_cache

        if attention_mask is None:
            attention_mask = jnp.ones((B, S), dtype=jnp.bool_)
        else:
            attention_mask = attention_mask.astype(jnp.bool_)

        if hasattr(self, "init_cache") and use_cache:
            self.init_cache(B, max_len)

        if key is None:
            temperature = 0.0
            key = jax.random.key(0)

        out_ids = jnp.zeros((B, max_len), dtype=jnp.int32)
        out_ids = out_ids.at[:, :S].set(input_ids.astype(jnp.int32))
        out_mask = jnp.zeros((B, max_len), dtype=jnp.bool_)
        out_mask = out_mask.at[:, :S].set(attention_mask)

        finished = jnp.zeros((B,), dtype=jnp.bool_)

        # Prefill on prompt
        prompt_position_ids = jnp.cumsum(attention_mask.astype(jnp.int32), axis=-1) - 1
        prompt_position_ids = jnp.maximum(prompt_position_ids, 0)
        logits = _forward_step(self, input_ids, attention_mask, prompt_position_ids)
        last_logit = logits[:, -1, :]  # shape (B, V)

        # Build suppress mask (tokens that should never be sampled)
        if suppress_tokens:
            suppress_mask = jnp.zeros(last_logit.shape[-1], dtype=jnp.bool_)
            for t in suppress_tokens:
                suppress_mask = suppress_mask.at[t].set(True)
        else:
            suppress_mask = None

        # Decode loop — each step dispatches a JIT-compiled forward pass to device
        for i in range(max_new_tokens):
            cur_logit = last_logit
            if suppress_mask is not None:
                cur_logit = jnp.where(suppress_mask, -1e9, cur_logit)

            next_token, key = sample_token(
                cur_logit, out_ids, out_mask,
                key, temperature, top_k, top_p, repetition_penalty
            )

            if eos_token_id is not None:
                next_token = jnp.where(finished, eos_token_id, next_token)
                finished = finished | (next_token == eos_token_id)

            next_token_2d = jnp.expand_dims(next_token.astype(jnp.int32), axis=1)

            # Write generated token and mark mask as valid
            out_ids = out_ids.at[:, S + i].set(next_token.astype(jnp.int32))
            out_mask = out_mask.at[:, S + i].set(True)

            # Masking for next forward pass
            indices = jnp.arange(max_len)
            valid_mask = indices <= (S + i)
            static_mask = out_mask & valid_mask

            if use_cache:
                decode_pos = jnp.expand_dims(
                    jnp.sum(static_mask.astype(jnp.int32), axis=-1) - 1, axis=-1
                )
                logits = _forward_step(self, next_token_2d, static_mask, decode_pos)
                last_logit = logits[:, -1, :]
            else:
                cur_pos = jnp.cumsum(static_mask.astype(jnp.int32), axis=-1) - 1
                cur_pos = jnp.maximum(cur_pos, 0)
                logits = _forward_step(self, out_ids, static_mask, cur_pos)
                last_logit = jax.lax.dynamic_slice(
                    logits, (0, S + i, 0), (B, 1, logits.shape[-1])
                )[:, 0, :]

        return out_ids

