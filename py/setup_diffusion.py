import torch
import torch.nn as nn
from tqdm import tqdm

# --- Configuration ---
IMG_SIZE = 64
BATCH_SIZE = 1 # For inference benchmark
TIMESTEPS = 200 # Number of diffusion steps
NOISE_DIM = 100 # Not used by U-Net directly, but for context
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===================================================================
# 1. DEFINE THE U-NET MODEL ARCHITECTURE
# A simple but representative U-Net for a diffusion model
# ===================================================================
class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        h = self.bn1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(...,) + (None,) * 2]
        h = h + time_emb
        h = self.bn2(self.relu(self.conv2(h)))
        return self.transform(h)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0, device=time.device)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=time.device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        image_channels = 3
        down_channels = (64, 128, 256, 512)
        up_channels = (512, 256, 128, 64)
        time_emb_dim = 32

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], time_emb_dim) for i in range(len(down_channels)-1)])
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], time_emb_dim, up=True) for i in range(len(up_channels)-1)])
        self.output = nn.Conv2d(up_channels[-1], image_channels, 1)

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        x = self.conv0(x)
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
        return self.output(x)

# ===================================================================
# 2. TRAIN THE MODEL FOR A FEW STEPS (IMPORTANT FOR STABILITY)
# ===================================================================
print("--- Training a small model for demonstration ---")
model = Unet().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
# Create fake data for a few training steps
dummy_dataloader = [(torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE, device=DEVICE), None) for _ in range(10)]

for epoch in range(2): # Just 2 epochs is enough
    for step, (batch, _) in enumerate(tqdm(dummy_dataloader, desc=f"Epoch {epoch+1}")):
        optimizer.zero_grad()
        t = torch.randint(0, TIMESTEPS, (BATCH_SIZE,), device=DEVICE).long()
        noise_pred = model(batch, t.float())
        loss = loss_fn(noise_pred, batch)
        loss.backward()
        optimizer.step()
print("Training complete.")

# ===================================================================
# 3. EXPORT THE REQUIRED ASSETS
# ===================================================================
model.eval()

# --- Export TorchScript model for LibTorch ---
output_file_pt = "unet_traced.pt"
print(f"\n--- Exporting TorchScript model to {output_file_pt} ---")
try:
    # U-Net has two inputs: the noisy image and the timestep
    example_img = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    example_time = torch.tensor([50.0], device=DEVICE) # Example time step
    
    traced_model = torch.jit.trace(model, (example_img, example_time))
    traced_model.save(output_file_pt)
    print("TorchScript model saved successfully.")
except Exception as e:
    print(f"Error tracing model: {e}")

# --- Export ONNX model for TensorRT ---
output_file_onnx = "unet.onnx"
print(f"\n--- Exporting ONNX model to {output_file_onnx} ---")
try:
    torch.onnx.export(model, (example_img, example_time), output_file_onnx,
                      input_names=['noisy_image', 'time_step'],
                      output_names=['predicted_noise'],
                      opset_version=13, # Use a reasonably modern opset
                      dynamic_axes={
                          'noisy_image': {0: 'batch_size'},
                          'time_step': {0: 'batch_size'},
                          'predicted_noise': {0: 'batch_size'}
                      })
    print("ONNX model saved successfully.")
except Exception as e:
    print(f"Error exporting ONNX: {e}")

print("\nAll setup complete.")