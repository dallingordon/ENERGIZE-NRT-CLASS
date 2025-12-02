import matplotlib.pyplot as plt  # For visualization
from PIL import Image  # Image loading and processing from files

from torch.utils.data import TensorDataset, DataLoader  # PyTorch data utilities
import torchvision.transforms as T  # Common image transforms
import torch.optim as optim  # Optimizers for training
import torch.nn as nn  # Neural network layers and loss functions
import torch  # PyTorch core library

from scipy.ndimage import gaussian_filter  # Gaussian smoothing filter
import numpy as np  # Array and numerical operations
import random  # Random number generation
import os  # File path utilities


def filter_empty_labels(images, labels):
    # Filter out any image-label pairs where the label is completely zero
    filtered_images = []
    filtered_labels = []

    # Iterate over images and labels simultaneously
    for img, lbl in zip(images, labels):
        lbl_np = np.array(lbl)  # Convert label to numpy array if not already
        # Check if label mask is not all zeros
        if not (lbl_np.min() == 0 and lbl_np.max() == 0):
            filtered_images.append(img)  # Keep image
            filtered_labels.append(lbl)  # Keep label

    # Convert filtered images list back to numpy array and return along with labels list
    return np.array(filtered_images), filtered_labels


def visualize_fake_data(x, y, start=0):
    # Visualize a 3x3 grid of synthetic images with their class titles
    fig, axs = plt.subplots(3, 3, figsize=(6, 6))
    for i in range(3):
        for j in range(3):
            idx = i * 3 + j + start  # Calculate data index to show
            axs[i, j].imshow(x[idx], cmap='gray')  # Show grayscale image
            axs[i, j].axis('off')  # Remove axis ticks and labels
            axs[i, j].set_title(['Shale', 'Sandstone', 'Mudstone'][y[idx]])  # Set title from labels
    plt.tight_layout()  # Adjust spacing
    plt.show()  # Display the plot


def visualize_fake_labels(x, labels, idx):
    import matplotlib.colors as mcolors

    cmap = mcolors.ListedColormap(['blue', 'cyan'])  # Define color map for labels

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))  # Two side-by-side plots

    axes[0].imshow(x[idx], cmap='gray')  # Show original grayscale image
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(labels[idx], cmap=cmap, interpolation='nearest')  # Show label with discrete colors
    axes[1].set_title('Color-coded Labels')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


def generate_shale_image(size=128):
    img = np.zeros((size, size))  # Initialize empty grayscale image
    num_grains = random.randint(10, 20)  # Random number of grains
    for _ in range(num_grains):
        x, y = np.random.randint(0, size, 2)  # Random center position
        radius = np.random.randint(size // 20, size // 10)  # Random grain radius
        y_grid, x_grid = np.ogrid[:size, :size]  # Create coordinate grid
        mask = (x_grid - x) ** 2 + (y_grid - y) ** 2 <= radius ** 2  # Circular mask
        img[mask] = 1  # Set mask pixels to white
    img = gaussian_filter(img, sigma=1)  # Smooth edges with Gaussian blur
    return img


def generate_sandstone_image(size=128):
    img = np.zeros((size, size))  # Initialize empty image
    num_stripes = random.randint(5, 15)  # Number of stripes
    thickness = random.randint(2, 5)  # Thickness of each stripe
    for _ in range(num_stripes):
        start = np.random.randint(0, size)  # Random start coordinate
        length = random.randint(size // 2, size)  # Random stripe length
        orientation = random.choice(['horizontal', 'vertical'])  # Random orientation

        if orientation == 'horizontal':
            # Check that stripes fit inside image boundaries
            if size - length <= 0 or size - thickness <= 0:
                print("Invalid parameters: size too small for length or thickness")
            else:
                # Choose random valid start point for stripe
                start = np.random.randint(0, size - length)
                end = start + length
                y = np.random.randint(0, size - thickness)
                img[y:y + thickness, start:end] = 1  # Draw horizontal stripe
        else:
            # Vertical stripe handling
            if size - length <= 0 or size - thickness <= 0:
                print("Invalid parameters: size too small for length or thickness")
            else:
                start = np.random.randint(0, size - length)
                end = start + length
                x = np.random.randint(0, size - thickness)
                img[start:end, x:x + thickness] = 1  # Draw vertical stripe
    img = gaussian_filter(img, sigma=1)  # Smooth edges
    return img


def generate_mudstone_image(size=128):
    img = np.zeros((size, size))  # Initialize empty image
    num_blobs = random.randint(3, 7)  # Number of overlapping blobs
    for _ in range(num_blobs):
        x, y = np.random.randint(0, size, 2)  # Random blob center
        radius = np.random.randint(size // 8, size // 4)  # Blob radius
        y_grid, x_grid = np.ogrid[:size, :size]
        mask = (x_grid - x) ** 2 + (y_grid - y) ** 2 <= radius ** 2  # Circular mask
        img[mask] = img[mask] + 1  # Stacking blobs (intensity accumulation)
    img = np.clip(img, 0, 1)  # Limit values to 1
    img = gaussian_filter(img, sigma=2)  # Smooth blobs heavily
    return img


def create_dataset_per_class(num_samples=100):
    X, y = [], []
    # Generate shale images and label 0
    for _ in range(num_samples):
        X.append(generate_shale_image())
        y.append(0)
    # Generate sandstone images and label 1
    for _ in range(num_samples):
        X.append(generate_sandstone_image())
        y.append(1)
    # Generate mudstone images and label 2
    for _ in range(num_samples):
        X.append(generate_mudstone_image())
        y.append(2)
    return np.array(X), np.array(y)


def label_image(img):
    # Determine threshold: 0.05 if floating point image, else 128 for uint8
    thresh = 0.05 if img.max() <= 1 else 128
    # Create binary mask by thresholding
    label = (img > thresh).astype(int)
    return label


# Define a simple encoder-decoder CNN for segmentation
class SimpleSegCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder: two convolutional layers + max pooling
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),  # Input channels=1, output=16, 3x3 conv
            nn.ReLU(),  # Activation
            nn.Conv2d(16, 32, 3, padding=1),  # Conv layer 2: 32 output channels
            nn.ReLU(),
            nn.MaxPool2d(2)  # Downsample by factor 2
        )
        # Decoder: transposed conv for upsampling and final conv layer with sigmoid
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 2, stride=2),  # Upsample by factor 2
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),  # Final conv: output 1 channel
            nn.Sigmoid()  # Maps output to [0,1] probability
        )

    def forward(self, x):
        x = self.encoder(x)  # Encode input image
        x = self.decoder(x)  # Decode features to mask
        return x


# Generate synthetic dataset with 50 samples per class
X, y = create_dataset_per_class(50)
print('Dataset size:', X.shape, y.shape)

# Generate label masks for each synthetic image
labels = [label_image(im) for im in X]

# Filter out images with empty labels
X, labels = filter_empty_labels(X, labels)

# Convert label list to numpy array
labels = np.array(labels)

# Keep a copy of original images if needed
X_org = X.copy()

# Convert images and labels to torch tensors, add channel dimension
X = torch.from_numpy(X).float().unsqueeze(1)  # Shape: (N,1,H,W)
Y = torch.from_numpy(labels).float().unsqueeze(1)  # Shape: (N,1,H,W)

# Create tensor dataset and dataloader for batching
dataset = TensorDataset(X, Y)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Initialize CNN model
model = SimpleSegCNN()

# Define binary cross-entropy loss function for segmentation
criterion = nn.BCELoss()
# Use Adam optimizer with learning rate 0.001
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop over 50 epochs
for epoch in range(50):
    model.train()  # Set model to training mode
    total_loss = 0
    for batch_x, batch_y in loader:
        optimizer.zero_grad()  # Reset gradients
        outputs = model(batch_x)  # Forward pass
        loss = criterion(outputs, batch_y)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update parameters
        total_loss += loss.item() * batch_x.size(0)  # Sum batch loss
    print(f'Epoch {epoch + 1}, Loss: {total_loss / len(dataset):.4f}')  # Average loss per sample

model.eval()  # Set model to evaluation mode
num_samples = 4  # Number of samples to visualize
indices = np.random.choice(len(dataset), num_samples, replace=False)  # Random indices

# Prepare discrete colormap for label visualization
import matplotlib.colors as mcolors

cmap = mcolors.ListedColormap(['blue', 'cyan'])

# Visualization of sample predictions vs true labels
with torch.no_grad():
    fig, axes = plt.subplots(num_samples, 2, figsize=(6, 3 * num_samples))
    for i, idx in enumerate(indices):
        img, true_label = dataset[idx]  # Get image and ground truth label
        img_input = img.unsqueeze(0)  # Add batch dim: (1,1,H,W)
        pred_prob = model(img_input)[0, 0]  # Predicted mask, shape (H,W)
        pred_label = (pred_prob >= 0.1).cpu().numpy()  # Threshold at 0.1

        # Convert tensors to numpy arrays for plotting
        true_label_np = true_label[0].cpu().numpy()  # Shape (H,W)

        # Left subplot: true labels
        ax_true = axes[i, 0] if num_samples > 1 else axes[0]
        ax_true.imshow(true_label_np, cmap=cmap, interpolation='nearest')
        ax_true.set_title('True Label')
        ax_true.axis('off')

        # Right subplot: predicted labels
        ax_pred = axes[i, 1] if num_samples > 1 else axes[1]
        ax_pred.imshow(pred_label, cmap=cmap, interpolation='nearest')
        ax_pred.set_title('Predicted Label')
        ax_pred.axis('off')

    plt.tight_layout()
    plt.show()

# Define paths relative to script location for experimental data
script_dir = os.path.dirname(os.path.abspath(__file__))
raw_path = os.path.join(script_dir, 'exp_data', 'imgs', 'plug_44_image_13-0_0.17um_top_right_up_left_123.png')
mask_path = os.path.join(script_dir, 'exp_data', 'masks', 'plug_44_image_13-0_0.17um_top_right_up_left_123_mask.png')

# Load experimental raw and mask images as grayscale
raw_img = Image.open(raw_path).convert('L')
mask_img = Image.open(mask_path).convert('L')

# Convert images to numpy arrays and normalize raw image to [0,1]
raw_arr = np.array(raw_img).astype(np.float32) / 255.0
mask_arr = np.array(mask_img)

# Convert mask to binary label assuming white=foreground (1), black=background (0)
label_arr = (mask_arr > 128).astype(np.float32)

# Convert raw image to tensor and add batch dimension
transform = T.Compose([T.ToTensor()])
raw_tensor = transform(raw_img).unsqueeze(0)  # Shape: (1,1,H,W)
label_tensor = torch.from_numpy(label_arr).unsqueeze(0).unsqueeze(0)  # Shape: (1,1,H,W)

# Predict mask on experimental raw image
model.eval()
with torch.no_grad():
    pred = model(raw_tensor)  # Output probability mask
    pred_label = (pred > 0.5).float()  # Threshold to binary mask

# Visualize raw image, true mask, and predicted mask side by side
plt.subplot(1, 3, 1)
plt.title('Raw Image')
plt.imshow(raw_arr, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('True Mask')
plt.imshow(label_arr, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Predicted Mask')
plt.imshow(pred_label.squeeze().cpu(), cmap='gray')
plt.axis('off')

plt.show()