import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# Parameters
L = 25               # grid size
dx = 1.0             # grid spacing
dt = 0.001           # time step (adjusted for typical dynamics)
M = 0.1              # mobility
epsilon = 1.0        # gradient energy coeff
n_steps = 100        # animation steps
benchmark_steps = 50 # steps for benchmarking
np.random.seed(42)

'''

Adjust these parameters to fine-tune your ML model training

'''
rollout_steps = 20   # rollout length for multi-step training
num_rollouts = 250   # number of rollouts for training
pde_correction_interval = 5  # apply PDE correction every N ML steps during rollout
n_epochs = 50 # number of training epochs for ML model



# Initialize phase field with grains
phi = 0.1 * np.random.randn(L, L)
for _ in range(5):
    x0, y0 = np.random.randint(5, L-5, 2)
    r = 5
    for x in range(L):
        for y in range(L):
            if (x - x0)**2 + (y - y0)**2 < r**2:
                phi[x, y] = 1.0 if np.random.rand() > 0.5 else -1.0

# PDE helpers
def laplacian(field):
    return (-4*field + np.roll(field, 1, 0) + np.roll(field, -1, 0) +
            np.roll(field, 1, 1) + np.roll(field, -1, 1)) / dx**2

def chemical_potential(field):
    return -epsilon**2 * laplacian(field) + 4 * field * (field**2 - 1)

def pde_step(field, dt):
    mu = chemical_potential(field)
    return field - dt * M * mu

# Residual block with BatchNorm
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

# Enhanced CNN: predicts increment delta_phi
class EnhancedCNN(nn.Module):
    def __init__(self, in_ch=1, channels=64, num_blocks=5):
        super().__init__()
        self.input_conv = nn.Conv2d(in_ch, channels, 3, padding=1)
        self.bn_in = nn.BatchNorm2d(channels)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]
        )
        self.output_conv = nn.Conv2d(channels, in_ch, 3, padding=1)
    def forward(self, x):
        out = F.relu(self.bn_in(self.input_conv(x)))
        out = self.res_blocks(out)
        out = self.output_conv(out)
        # No tanh here, increments can be positive or negative
        return out

# Normalize to [0,1]
def normalize_data(x):
    return (x + 1) / 2

# Denormalize to [-1,1]
def denormalize_data(x):
    return 2 * x - 1

# Generate multi-step rollout training data
def generate_training_data_multistep(phi_init, steps, rollouts):
    data = []
    for _ in range(rollouts):
        phi_curr = phi_init.copy()
        rollout_seq = [phi_curr]
        for _ in range(steps):
            phi_next = pde_step(phi_curr, dt)
            rollout_seq.append(phi_next)
            phi_curr = phi_next
        data.append(np.array(rollout_seq))
    return np.array(data)  # shape (rollouts, steps+1, L, L)

# Prepare inputs/targets: inputs are phi^n, targets are increment delta_phi = phi^{n+1} - phi^n
def prepare_inputs_targets_multistep(data):
    inputs = data[:, :-1, :, :]
    targets = data[:, 1:, :, :] - inputs
    r, s = inputs.shape[:2]
    inputs = inputs.reshape((r*s, L, L))
    targets = targets.reshape((r*s, L, L))
    inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(1)
    targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
    return inputs, targets

# Training function with noise injection and multi-step batches
def train_model_multistep_noise(model, optimizer, criterion, inputs, targets,
                               epochs=25, batch_size=64, device='cpu',
                               noise_std=0.03):
    dataset_size = inputs.shape[0]
    indices = np.arange(dataset_size)

    model.train()
    for epoch in range(epochs):
        np.random.shuffle(indices)
        total_loss = 0.0
        for start in range(0, dataset_size, batch_size):
            batch_idx = indices[start:start+batch_size]
            batch_in = inputs[batch_idx].to(device)
            batch_tgt = targets[batch_idx].to(device)

            # Inject Gaussian noise in inputs
            noisy_in = batch_in + torch.randn_like(batch_in) * noise_std
            noisy_in = torch.clamp(noisy_in, 0.0, 1.0)  # keep normalized range

            pred = model(noisy_in)
            loss = criterion(pred, batch_tgt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_idx)

        avg_loss = total_loss / dataset_size
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

# Instantiate device and model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EnhancedCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Generate & prepare training data
print("Generating rollout training data...")
rollout_data = generate_training_data_multistep(phi, rollout_steps, num_rollouts)
rollout_data_norm = normalize_data(rollout_data)

inputs_train, targets_train = prepare_inputs_targets_multistep(rollout_data_norm)

print("Training with noise injection...")
train_model_multistep_noise(model, optimizer, criterion, inputs_train,
                           targets_train, epochs=n_epochs, batch_size=32,
                           device=device, noise_std=0.1)

# Simulation plus hybrid rollout: ML prediction plus PDE corrections every N steps
phi_pde = phi.copy()
phi_ml = torch.tensor(phi, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

def ml_model_step(phi_ml):
    # Normalize input, predict increment, then update and denormalize
    phi_ml_norm = normalize_data(phi_ml)
    with torch.no_grad():
        delta_norm = model(phi_ml_norm)
    delta = delta_norm  # increments not normalized, predicted in original range
    phi_next = phi_ml + delta
    return phi_next

fig, ax = plt.subplots(1, 3, figsize=(12,4))
im_pde = ax[0].imshow(phi_pde, cmap='RdBu', vmin=-1, vmax=1)
ax[0].set_title("PDE simulation")
im_ml = ax[1].imshow(phi_pde, cmap='RdBu', vmin=-1, vmax=1)
ax[1].set_title("ML accelerated + PDE corrections")
im_diff = ax[2].imshow(np.zeros_like(phi_pde), cmap='bwr', vmin=-0.5, vmax=0.5)
ax[2].set_title("Difference (ML-PDE)")
for a in ax:
    a.axis('off')

def update(frame):
    global phi_pde, phi_ml
    # PDE step
    phi_pde = pde_step(phi_pde, dt)
    im_pde.set_data(phi_pde)

    # Hybrid ML-PDE rollout
    if frame % pde_correction_interval == 0:
        # apply PDE correction step every N steps
        phi_ml_np = phi_ml.cpu().numpy().squeeze()
        phi_ml_np = pde_step(phi_ml_np, dt)
        phi_ml = torch.tensor(phi_ml_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    else:
        # ML predict increment and update
        phi_ml = ml_model_step(phi_ml)

    im_ml.set_data(phi_ml.cpu().squeeze().numpy())
    diff = phi_ml.cpu().squeeze().numpy() - phi_pde
    im_diff.set_data(diff)

    ax[0].set_title(f"PDE simulation: step {frame}")
    ax[1].set_title(f"ML accelerated + PDE corrections: step {frame}")
    ax[2].set_title(f"Difference (ML - PDE): step {frame}")

    return [im_pde, im_ml, im_diff]

ani = FuncAnimation(fig, update, frames=n_steps, interval=50, blit=False)
plt.tight_layout()
plt.show()

# Benchmark speed and accuracy
def rmse(a, b):
    return np.sqrt(np.mean((a - b)**2))

phi_pde_bench = phi.copy()
phi_ml_bench = torch.tensor(phi, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

pde_fields = []
start_pde = time.time()
for _ in range(benchmark_steps):
    phi_pde_bench = pde_step(phi_pde_bench, dt)
    pde_fields.append(phi_pde_bench.copy())
end_pde = time.time()
pde_time = end_pde - start_pde

ml_fields = []
start_ml = time.time()
model.eval()
with torch.no_grad():
    for i in range(benchmark_steps):
        if i % pde_correction_interval == 0:
            phi_ml_np = phi_ml_bench.cpu().numpy().squeeze()
            phi_ml_np = pde_step(phi_ml_np, dt)
            phi_ml_bench = torch.tensor(phi_ml_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        else:
            phi_ml_bench = ml_model_step(phi_ml_bench)
        ml_fields.append(phi_ml_bench.cpu().squeeze().numpy())
end_ml = time.time()
ml_time = end_ml - start_ml

rmses = [rmse(pde_fields[i], ml_fields[i]) for i in range(benchmark_steps)]
mean_rmse = np.mean(rmses)
speedup = pde_time / ml_time if ml_time > 0 else float('inf')

print(f"--- Benchmark results after training ---")
print(f"PDE time for {benchmark_steps} steps: {pde_time:.4f} s")
print(f"ML (hybrid) time for {benchmark_steps} steps: {ml_time:.4f} s")
print(f"Mean RMSE between PDE and ML: {mean_rmse:.6f}")
print(f"Speedup factor (PDE / ML): {speedup:.2f}x")
print("\nRun multiple simulations with different params and record Mean RMSE and Speedup for comparison.")