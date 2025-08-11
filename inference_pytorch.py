import torch
import time
NUM_ITERATIONS = 10
WARMUP_ITERATIONS = 2
BATCH_SIZE = 1
TIMESTEPS = 50 # Use fewer steps for a faster benchmark
IMG_SIZE = 64
DEVICE = "cuda"

if __name__ == "__main__":
    print("--- Performance Test: Diffusion with Python/PyTorch ---")
    model = torch.jit.load("unet_traced.pt").to(DEVICE)
    model.eval()

    # Pre-calculate scheduler constants
    betas = torch.linspace(0.0001, 0.02, TIMESTEPS, device=DEVICE)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)

    print(f"Batch: {BATCH_SIZE}, Denoising Steps: {TIMESTEPS}, Iterations: {NUM_ITERATIONS}")

    # Warm-up
    print("Warming up...")
    with torch.no_grad():
        for _ in range(WARMUP_ITERATIONS):
            img = torch.randn((BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE), device=DEVICE)
            for i in reversed(range(TIMESTEPS)):
                t = torch.full((BATCH_SIZE,), i, device=DEVICE, dtype=torch.float)
                predicted_noise = model(img, t)
    torch.cuda.synchronize()

    print("Starting benchmark...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    with torch.no_grad():
        for _ in range(NUM_ITERATIONS):
            img = torch.randn((BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE), device=DEVICE)
            for i in reversed(range(TIMESTEPS)):
                t = torch.full((BATCH_SIZE,), i, device=DEVICE, dtype=torch.float)
                predicted_noise = model(img, t)
                # This is a simplified DDPM sampling step
                alpha_t = alphas[i]
                alpha_t_cumprod = alphas_cumprod[i]
                beta_t = betas[i]
                if i > 0:
                    noise = torch.randn_like(img)
                else:
                    noise = torch.zeros_like(img)
                
                img = 1 / torch.sqrt(alpha_t) * (img - ((1 - alpha_t) / (torch.sqrt(1 - alpha_t_cumprod))) * predicted_noise) + torch.sqrt(beta_t) * noise

    end_event.record()
    torch.cuda.synchronize()

    total_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = total_time_ms / NUM_ITERATIONS

    print("\n--- PyTorch Results ---")
    print(f"Total time for {NUM_ITERATIONS} generations: {total_time_ms:.3f} ms")
    print(f"Average time per generation ({TIMESTEPS} steps): {avg_time_ms:.4f} ms")