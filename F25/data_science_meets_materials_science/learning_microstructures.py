#!/usr/bin/env python3
"""
Synthetic EBSD-like microstructure pipeline with watershed post-processing:
- Generate N synthetic EBSD maps (Voronoi tessellation).
- Train/test a U-Net on grain boundary segmentation.
- Apply watershed to separate predicted grains.
- Plot ground truth vs predicted grain labels side by side.
"""

import os, random, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from scipy.spatial import Voronoi
from skimage import morphology, measure
from skimage.segmentation import watershed

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


# -----------------------------
# Synthetic Data Generator
# -----------------------------
def generate_voronoi_grains(size=128, n_grains=30, seed=None):
    if seed: np.random.seed(seed)
    pts = np.random.rand(n_grains, 2) * size
    xv, yv = np.meshgrid(np.arange(size), np.arange(size))
    coords = np.stack([xv.ravel(), yv.ravel()], axis=-1)

    dists = ((coords[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
    labels = dists.argmin(axis=1).reshape(size, size)

    orientations = np.random.rand(n_grains, 3) * 2 * np.pi
    euler_map = orientations[labels]

    return labels, euler_map


def generate_synthetic_dataset(n_samples=20, size=128, out_dir="./synthetic_data"):
    os.makedirs(out_dir, exist_ok=True)
    for i in range(n_samples):
        labels, euler = generate_voronoi_grains(size=size, n_grains=random.randint(10, 40))
        boundary = np.zeros_like(labels, dtype=np.uint8)
        boundary[:-1, :] |= (labels[:-1, :] != labels[1:, :])
        boundary[:, :-1] |= (labels[:, :-1] != labels[:, 1:])

        sample_dir = os.path.join(out_dir, f"sample_{i:03d}")
        os.makedirs(sample_dir, exist_ok=True)
        np.save(os.path.join(sample_dir, "grain_ids.npy"), labels)
        np.save(os.path.join(sample_dir, "ebsd_euler.npy"), euler)
        np.save(os.path.join(sample_dir, "boundary_mask.npy"), boundary)


# -----------------------------
# Dataset
# -----------------------------
class SyntheticEBSDDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        for s in sorted(os.listdir(root_dir)):
            sd = os.path.join(root_dir, s)
            if os.path.isdir(sd):
                self.samples.append(sd)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sd = self.samples[idx]
        euler = np.load(os.path.join(sd, "ebsd_euler.npy"))
        boundary = np.load(os.path.join(sd, "boundary_mask.npy"))
        euler = torch.from_numpy(euler).permute(2, 0, 1).float() / np.pi
        boundary = torch.from_numpy(boundary).unsqueeze(0).float()
        return euler, boundary, sd


# -----------------------------
# Model (U-Net)
# -----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

    '''
    
    TO-DO

    '''

    def forward(self, x):
        return self.seq(x)


class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        self.d1 = DoubleConv(in_ch, 32)
        self.d2 = DoubleConv(32, 64)
        self.d3 = DoubleConv(64, 128)

        self.u1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.u2 = nn.ConvTranspose2d(64, 32, 2, stride=2)

        self.d2_up = DoubleConv(128, 64)
        self.d1_up = DoubleConv(64, 32)
        self.outc = nn.Conv2d(32, out_ch, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        '''
    
        TO-DO

        '''
        return x


# -----------------------------
# Training
# -----------------------------
def train_model(model, train_loader, val_loader, device, epochs=20, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        '''
    
        TO-DO

        '''
        print(f"Epoch {ep + 1}: train {tr_loss / len(train_loader):.4f}, val {vl / len(val_loader):.4f}")


# -----------------------------
# Watershed Post-processing
# -----------------------------
def watershed_postprocess(prob_mask, thresh=0.075):
    """
    Postprocess CNN probability map into grain segments using watershed.
    - prob_mask: 2D numpy array, CNN boundary probability [0,1]
    - thresh: threshold for defining grain interior (default 0.5)
    """
    # Grains are the opposite of the boundary probability
    grain_mask = prob_mask < thresh  

    # Distance transform inside grains
    distance = distance_transform_edt(grain_mask)

    # Find local maxima inside grains for watershed seeds
    local_maxi = morphology.local_maxima(distance)
    markers = measure.label(local_maxi)

    # Apply watershed segmentation
    labeled = watershed(-distance, markers, mask=grain_mask)
    return labeled


# -----------------------------
# Postprocess & Visualization
# -----------------------------
def predict_and_plot(model, dataloader, device, out_dir, brighten=2.0):
    """
    Plot CNN probability map vs ground truth labels vs watershed segmented + labeled grains.
    """
    model.eval()
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            if isinstance(data, (list, tuple)):
                images, labels = data[:2]
            else:
                images = data
                labels = None

            images = images.to(device)
            outputs = model(images)
            probs_batch = torch.sigmoid(outputs).squeeze(1).cpu().numpy()  # [B,H,W]

            for i in range(probs_batch.shape[0]):
                probs = probs_batch[i]
                bright_probs = np.clip(probs * brighten, 0, 1)

                # --- Watershed segmentation ---
                labeled_grains = watershed_postprocess(probs)
                props = measure.regionprops(labeled_grains)

                # --- Create figure with 3 subplots ---
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                # 1) CNN probability map
                axes[0].imshow(bright_probs, cmap='gray')
                axes[0].set_title("CNN Probability Map")
                axes[0].axis('off')

                # 2) Ground truth labels
                if labels is not None:
                    label_img = labels[i].cpu().numpy()
                    if label_img.ndim == 3 and label_img.shape[0] == 1:
                        label_img = label_img.squeeze(0)
                    axes[1].imshow(label_img, cmap='tab20')
                    axes[1].set_title("Ground Truth")
                axes[1].axis('off')

                # 3) Watershed segmented grains with labels
                axes[2].imshow(labeled_grains, cmap='nipy_spectral')
                axes[2].set_title(f"Watershed Grains (count={len(props)})")
                axes[2].axis('off')

                 # Add red numbers at grain centroids
                for prop in props:
                    y, x = prop.centroid
                    axes[2].text(x, y, str(prop.label), color='red',
                                 ha='center', va='center', fontsize=8)

                # Add red labels at grain centroids
                for prop in props:
                    y, x = prop.centroid
                    axes[2].text(x, y, str(prop.label), color='red', ha='center', va='center', fontsize=8)

                plt.tight_layout()
                out_path = f"{out_dir}/compare_{idx}_{i}.png"
                plt.savefig(out_path, dpi=150)
                plt.show()

# -----------------------------
# Main
# -----------------------------
def main():
    class Args:
        synthetic = 50
        size = 128
        epochs = 50
        batch = 4
        out_dir = "./results"

    if "ipykernel" in sys.modules:
        args = Args()
    else:
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("--synthetic", type=int, default=20)
        ap.add_argument("--size", type=int, default=128)
        ap.add_argument("--epochs", type=int, default=20)
        ap.add_argument("--batch", type=int, default=4)
        ap.add_argument("--out_dir", type=str, default="./results")
        args = ap.parse_args()

    data_dir = os.path.join(args.out_dir, "synthetic_data")
    generate_synthetic_dataset(args.synthetic, size=args.size, out_dir=data_dir)

    ds = SyntheticEBSDDataset(data_dir)
    n_train = int(0.8 * len(ds))
    n_val = len(ds) - n_train
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(in_ch=3, out_ch=1).to(device)

    train_model(model, train_loader, val_loader, device, epochs=args.epochs)
    predict_and_plot(model, val_loader, device, args.out_dir, brighten=5.0)


if __name__ == "__main__":
    main()
