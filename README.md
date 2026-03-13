# Zlynx

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/zlynx-ai/zlynx)

Zlynx is a lightweight, highly-customizable deep learning library built on top of **JAX** and **Flax NNX**. It is designed for researchers and developers who want fine-grained control over model architectures, training loops, and distributed setups without the bloat of massive frameworks.

## Quick Start

```bash
pip install zlynx
```

### Define your model and load weights

```python
from zlynx import Z

# Define your model architecture
class MyModel(Z): ...

# Load weights from HuggingFace
model, tokenizer = MyModel.load_hf("username/my-model")

# Load from Kaggle
import kagglehub
kagglehub.login()
model, _ = MyModel.load_kaggle("username/model", sharding="fsdp")
```

### Sharding (Optional)

Sharding distributes your model across multiple GPUs/TPUs. Omit for single-device loading:

```python
# Single device (default)
model, tokenizer = MyModel.load_hf("username/my-model")

# Distributed across devices
model, _ = MyModel.load_hf("username/my-model", sharding="fsdp")  # shards model weights
model, _ = MyModel.load_hf("username/my-model", sharding="ddp")   # shards data
```

### Or use built-in Llama

```python
from zlynx.models.llama import LlamaConfig, LlamaLanguageModel

config = LlamaConfig(vocab_size=32000, hidden_size=512, num_hidden_layers=2)
model = LlamaLanguageModel(config)
```

### Save & Push

```python
model.save("my-model", format="safetensors")
model.push_hf("username/my-model", private=False)
```

### Train

```python
from zlynx import Trainer, TrainerConfig
trainer = Trainer(model, train_dataset, TrainerConfig())
trainer.train()
```
