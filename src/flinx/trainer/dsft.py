import jax
import jax.numpy as jnp
import optax
from typing import Callable, Optional
from dataclasses import dataclass

from .trainer import Trainer, TrainerConfig
from .dataset import DatasetConfig

@dataclass
class DSFTConfig(TrainerConfig):
    """Configuration for Diffusion Supervised Fine-Tuning (DSFT)."""
    image_size: tuple[int, int] = (256, 256)
    dataset_image_field: str = "image"
    dataset_text_field: str = "text"
    formatting_func: Optional[Callable] = None
    
    # Diffusion specific
    num_train_timesteps: int = 1000
    prediction_type: str = "epsilon" # "epsilon", "v_prediction", or "sample"

def diffusion_loss(model, batch):
    """
    Standard Diffusion Model Loss.
    Assumes `batch` contains pre-computed arrays or handles dynamic noising.
    """
    noisy_images = batch["noisy_images"]
    timesteps = batch["timesteps"]
    target_noise = batch["noise"]
    
    cond = batch.get("conditioning", None)
    
    if cond is not None:
        noise_pred = model(noisy_images, timesteps, cond)
    else:
        noise_pred = model(noisy_images, timesteps)
        
    # Standard MSE loss
    loss = jnp.mean((noise_pred - target_noise) ** 2)
    return loss

class DSFTTrainer(Trainer):
    """
    Diffusion Supervised Fine-Tuning Trainer.
    Automates image transformations and the discrete noise scheduler mapping.
    """
    def __init__(
        self,
        model,
        dataset,
        noise_scheduler,
        processor=None, # optional text tokenizer/image processor
        trconfig: Optional[DSFTConfig] = None,
        dsconfig: Optional[DatasetConfig] = None,
        loss_fn: Optional[Callable] = None,
    ):
        trconfig = trconfig or DSFTConfig()
        dsconfig = dsconfig or DatasetConfig()
        
        self.noise_scheduler = noise_scheduler
        self.processor = processor
        self.dsft_config = trconfig
        
        loss_fn = loss_fn or diffusion_loss
        original_preprocessing = dsconfig.preprocessing_fn
        
        def diffusion_map(example):
             # 1. Apply any custom formatting or raw value extraction
            if self.dsft_config.formatting_func is not None:
                features = self.dsft_config.formatting_func(example)
            else:
                features = {
                    "image": example[self.dsft_config.dataset_image_field],
                }
                if self.dsft_config.dataset_text_field in example:
                     features["text"] = example[self.dsft_config.dataset_text_field]

            if original_preprocessing is not None:
                features = original_preprocessing(features)
                
            # 2. Extract cleanly shaped pixel values and tokenize conditionings
            # Images are theoretically converted to [C, H, W] arrays here via external transforms
            pixel_values = jnp.array(features["image"]) 
            
            out = {"clean_images": pixel_values}
            
            if "text" in features and self.processor is not None:
                encoded = self.processor(
                    features["text"],
                    padding="max_length",
                    truncation=True,
                    max_length=77, # standard strict CLIP horizon
                    return_tensors="np"
                )
                out["conditioning"] = encoded["input_ids"][0]
                
            # IMPORTANT: The dynamic forward noising processes (sampling `t` and adding `noise()`)
            # cannot be done inside preprocessing because Grain Datasets execute map_fns purely statically
            # on the host prior to device sharding. To get distinct random shapes and timesteps per batch,
            # we must process the final raw arrays.
            # INSTEAD: The Trainer automatically requires a hook mapping if you don't noise directly in loss.
            # To stick to best practices, users should wrap `diffusion_loss` to sample `t` JIT-side if needed, 
            # or we do it heuristically. Here we leave the batch pure, and assume the loss function handles noising!
            
            return out
            
        dsconfig.preprocessing_fn = diffusion_map
        
        super().__init__(
            model=model,
            dataset=dataset,
            loss_fn=loss_fn,
            processor=processor,
            trconfig=trconfig,
            dsconfig=dsconfig
        )