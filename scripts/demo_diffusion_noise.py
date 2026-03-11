import numpy as np
import matplotlib.pyplot as plt
import os

def q_sample(x_start, t, alphas_cumprod, noise=None):
    if noise is None:
        noise = np.random.randn(*x_start.shape)
    
    sqrt_alphas_cumprod_t = np.sqrt(alphas_cumprod[t])
    sqrt_one_minus_alphas_cumprod_t = np.sqrt(1. - alphas_cumprod[t])
    
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

def main():
    timesteps = 1000
    beta_start = 0.0001
    beta_end = 0.02
    betas = np.linspace(beta_start, beta_end, timesteps)
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    # Create 64x64 geometric pattern (Pink and Cyan checkerboard)
    grid = np.zeros((64, 64, 3))
    for i in range(64):
        for j in range(64):
            if (i // 16) % 2 == (j // 16) % 2:
                grid[i, j] = [0.9, 0.1, 0.5] # Pink
            else:
                grid[i, j] = [0.1, 0.8, 0.9] # Cyan

    # Add a yellow circle in the middle
    x, y = np.meshgrid(np.linspace(-1, 1, 64), np.linspace(-1, 1, 64))
    d = np.sqrt(x*x + y*y)
    grid[d < 0.4] = [0.9, 0.9, 0.1] # Yellow

    # Scale pixel values [0, 1] to [-1, 1] for the diffusion math
    x_start = grid * 2.0 - 1.0

    steps_to_show = [0, 50, 100, 200, 400, 600, 800, 999]
    fig, axes = plt.subplots(1, len(steps_to_show), figsize=(16, 2.5))

    for i, t in enumerate(steps_to_show):
        if t == 0:
            noisy_img_disp = grid
        else:
            noise = np.random.randn(*x_start.shape)
            noisy_img = q_sample(x_start, t, alphas_cumprod, noise)
            
            # Rescale noisy image back to [0, 1] for display
            noisy_img_disp = (noisy_img + 1.0) / 2.0
            noisy_img_disp = np.clip(noisy_img_disp, 0, 1)
            
        axes[i].imshow(noisy_img_disp)
        axes[i].set_title(f"Step {t}")
        axes[i].axis('off')

    plt.tight_layout()
    
    # Save the output to the active artifacts directory so the Agent UI can render it
    save_path = '/home/shinapri/.gemini/antigravity/brain/0822f1cd-5f55-4cf5-81fa-4a0af8d0d9b4/diffusion_forward.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Diffusion sequence saved to: {save_path}")

if __name__ == '__main__':
    main()
