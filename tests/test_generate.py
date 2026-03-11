import jax
import jax.numpy as jnp
import time
from zlynx.models.llama import LlamaConfig, LlamaLanguageModel

def test_fast_generation():
    print("\n=== Testing JAX-Native High-Throughput Generation ===")
    
    # Small model for quick testing
    config = LlamaConfig(
        vocab_size=128, 
        hidden_size=64, 
        intermediate_size=128, 
        num_hidden_layers=2, 
        head_dim=16
    )
    
    # Initialize Model
    key = jax.random.key(42)
    model = LlamaLanguageModel(config, key=key)
    
    # Create a batch of dummy prompts
    batch_size = 8
    prompt_len = 16
    max_new_tokens = 32
    
    prompts = jnp.ones((batch_size, prompt_len), dtype=jnp.int32)
    
    print("\n1. Warming up JIT compiler...")
    start_compile = time.time()
    # The first run triggers the `@nnx.jit` compilation of the entire `while_loop`
    _ = model.generate(prompts, max_new_tokens=max_new_tokens, temperature=0.0)
    print(f"Compilation finished in {time.time() - start_compile:.2f} seconds.")
    
    print("\n2. Benchmarking execution speed (No JIT overhead)...")
    start_run = time.time()
    out = model.generate(prompts, max_new_tokens=max_new_tokens, temperature=0.0)
    elapsed = time.time() - start_run
    
    # Calculate tokens per second
    total_generated_tokens = batch_size * max_new_tokens
    tps = total_generated_tokens / elapsed
    
    print(f"Generated {total_generated_tokens} tokens in {elapsed:.4f} seconds.")
    print(f"Throughput: {tps:.2f} tokens/second.")
    
    assert out.shape == (batch_size, prompt_len + max_new_tokens)
    print("\nGeneration verified successfully!")

if __name__ == "__main__":
    test_fast_generation()
