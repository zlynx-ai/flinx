import jax
import jax.numpy as jnp
import optax
from typing import Callable, Optional
from collections.abc import Iterable

from .trainer import Trainer, TrainerConfig
from .dataset import DatasetConfig

from dataclasses import dataclass

@dataclass
class SFTConfig(TrainerConfig):
    """Configuration for Supervised Fine-Tuning (SFT)."""
    max_seq_len: int = 1024
    dataset_text_field: str = "text"
    formatting_func: Optional[Callable] = None


def causal_lm_loss(model, batch):
    """
    Standard autoregressive causal language modeling loss.
    Assumes `batch` has "input_ids" and (optionally) "labels" and "attention_mask".
    Delegates calculation natively to the model outputs.
    """
    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask", None)
    labels = batch.get("labels", input_ids)
    
    # Forward pass: calculates shifted cross-entropy internally
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    
    return outputs.loss


class SFTTrainer(Trainer):
    """
    Supervised Fine-Tuning Trainer.
    Automates dataset tokenization, padding, and truncation.
    """
    def __init__(
        self,
        model,
        dataset,
        processor,
        trconfig: Optional[SFTConfig] = None,
        dsconfig: Optional[DatasetConfig] = None,
        loss_fn: Optional[Callable] = None,
    ):
        trconfig = trconfig or SFTConfig()
        dsconfig = dsconfig or DatasetConfig()
        
        if processor is None:
            raise ValueError("SFTTrainer requires a `processor` (tokenizer) for dataset formatting.")
            
        self.processor = processor
        self.sft_config = trconfig
        
        # Default causal LM loss if none explicitly provided
        loss_fn = loss_fn or causal_lm_loss
        
        # Auto-inject or wrap the DatasetConfig preprocessing function
        original_preprocessing = dsconfig.preprocessing_fn
        
        def sft_tokenizer_map(example):
            # 1. Formatting
            if self.sft_config.formatting_func is not None:
                text = self.sft_config.formatting_func(example)
            else:
                text = example.get(self.sft_config.dataset_text_field, "")
                if not text and isinstance(example, dict):
                    # fallback if "text" field is missing but dict has one key
                    keys = list(example.keys())
                    if len(keys) == 1:
                        text = example[keys[0]]
            
            # Allow the user's preprocessing to mutate the text/example first if they want
            if original_preprocessing is not None:
                text = original_preprocessing(text)
                
            # 2. Tokenize, Truncate, Pad
            # We must enforce fixed lengths for JAX compiled loops
            encoded = self.processor(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.sft_config.max_seq_len,
                return_tensors="np" # JAX deals with numpy arrays before converting to jnp
            )
            
            # Strip the batch dimension [1, seq_len] -> [seq_len] that HF tokenizers add
            return {
                "input_ids": encoded["input_ids"][0],
                "attention_mask": encoded["attention_mask"][0]
            }
            
        dsconfig.preprocessing_fn = sft_tokenizer_map
        
        super().__init__(
            model=model,
            dataset=dataset,
            loss_fn=loss_fn,
            processor=processor,
            trconfig=trconfig,
            dsconfig=dsconfig
        )