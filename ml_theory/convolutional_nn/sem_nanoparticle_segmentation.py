"""


Purpose
-------
This example introduces a simple convolutional neural network (CNN) for
image segmentation using synthetic SEM-like images of crystalline
nanoparticles.

This second version improves the synthetic image generator so that the
particles do NOT overlap. The particles are placed in different regions
of the image using a simple rejection-sampling approach.

The script performs the following tasks:

    1. Generate a synthetic SEM-style dataset
    2. Save grayscale images and segmentation masks
    3. Train a small CNN to predict where nanoparticles are located
    4. Predict particle masks for test images
    5. Estimate particle-size statistics from the masks
    6. Create plots that overlay CNN-predicted particle outlines on the
       original SEM images
    7. Save diagnostic plots and tables

The script is intentionally simple and highly commented so that students can
follow the main ideas without needing prior experience in image analysis.

IMPORTANT
---------
The SEM images are synthetic. They are designed only for teaching.
"""

from pathlib import Path
import math
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import ndimage as ndi

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ===========================================================================
# 1. PATHS AND SETTINGS
# ===========================================================================

SCRIPT_DIRECTORY = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# LOCAL WORK DIRECTORY
# ---------------------------------------------------------------------------

# This script may itself be stored inside a synced Windows directory such as:
#
#     /mnt/c/Users/.../OneDrive/...
#
# when it is run through WSL.
#
# Large numbers of generated training images are better stored in the local
# Linux filesystem instead of a mounted Windows/OneDrive directory.
#
# Path.home() works on both:
#
#     WSL / Linux:   /home/username
#     macOS:         /Users/username
#
# The script can therefore stay inside the course Git repository while the
# generated SEM images and results are stored in a safe local working folder.

WORK_DIRECTORY = (
    Path.home()
    / "materials_by_design_course_data"
    / "cnn_sem_nanoparticle_segmentation"
)

DATASET_DIRECTORY = (
    WORK_DIRECTORY
    / "sem_nanoparticle_dataset_v2"
)

IMAGE_DIRECTORY = (
    DATASET_DIRECTORY
    / "images"
)

MASK_DIRECTORY = (
    DATASET_DIRECTORY
    / "masks"
)

RESULTS_DIRECTORY = (
    WORK_DIRECTORY
    / "cnn_results_v2"
)


# Create the directories if they do not already exist.

IMAGE_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)

MASK_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)


# Random seeds make the results reproducible.
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# This introductory example uses the CPU.
DEVICE = torch.device("cpu")


# Dataset settings.
NUMBER_OF_IMAGES = 180
IMAGE_SIZE = 128

# Number of training epochs.
NUMBER_OF_EPOCHS = 18

# Batch size used during CNN training.
BATCH_SIZE = 8

# Learning rate for the optimizer.
LEARNING_RATE = 0.001

# Fraction of images used for training.
TRAIN_FRACTION = 0.8

# Particle-size thresholds based on equivalent radius in pixels.
SMALL_RADIUS_MAX = 7.0
MEDIUM_RADIUS_MAX = 13.0

# Minimum gap between neighboring particles.
MINIMUM_PARTICLE_GAP = 2.0

# Maximum number of placement attempts for one particle.
MAX_PLACEMENT_ATTEMPTS = 200


print("=" * 75)
print("CNN for Synthetic SEM Nanoparticle Segmentation")
print("Second iteration: non-overlapping particles and outline overlays")
print("=" * 75)
print()
print("Local work directory:")
print(WORK_DIRECTORY)
print()
print("Dataset directory:")
print(DATASET_DIRECTORY)
print()
print("Results directory:")
print(RESULTS_DIRECTORY)
print()
print("Device:", DEVICE)
print()


# ===========================================================================
# 2. HELPER FUNCTIONS FOR SYNTHETIC IMAGE GENERATION
# ===========================================================================

def choose_particle_size_mode(image_index):
    """
    Assign one image to a specific size-mixture mode.

    Different modes create images with different populations of particle sizes.
    This makes the dataset more varied and more realistic for teaching.
    """

    modes = [
        "mostly_small",
        "mostly_medium",
        "mostly_large",
        "balanced",
        "small_large_mix",
        "broad_mixture",
    ]

    mode = modes[image_index % len(modes)]

    return mode


def sample_particle_radius(mode):
    """
    Sample a particle radius in pixels based on the image mode.
    """

    if mode == "mostly_small":
        radius = np.random.uniform(3.0, 7.0)

    elif mode == "mostly_medium":
        radius = np.random.uniform(7.0, 13.0)

    elif mode == "mostly_large":
        radius = np.random.uniform(13.0, 22.0)

    elif mode == "balanced":

        category = np.random.choice(
            ["small", "medium", "large"],
            p=[1/3, 1/3, 1/3]
        )

        if category == "small":
            radius = np.random.uniform(3.0, 7.0)
        elif category == "medium":
            radius = np.random.uniform(7.0, 13.0)
        else:
            radius = np.random.uniform(13.0, 22.0)

    elif mode == "small_large_mix":

        category = np.random.choice(
            ["small", "large"],
            p=[0.55, 0.45]
        )

        if category == "small":
            radius = np.random.uniform(3.0, 7.0)
        else:
            radius = np.random.uniform(13.0, 22.0)

    else:
        radius = np.random.uniform(3.0, 22.0)

    return radius


def classify_radius(radius):
    """
    Convert a particle radius into a size label.
    """

    if radius < SMALL_RADIUS_MAX:
        return "small"
    elif radius < MEDIUM_RADIUS_MAX:
        return "medium"
    else:
        return "large"


def particles_overlap(new_x, new_y, new_radius, placed_particles, minimum_gap):
    """
    Check whether a newly proposed particle overlaps a previously placed
    particle.

    placed_particles is a list of tuples:
        (center_x, center_y, radius)

    Two particles are considered too close if the distance between centers is
    smaller than the sum of their radii plus a required gap.
    """

    for old_x, old_y, old_radius in placed_particles:

        center_distance = math.sqrt(
            (new_x - old_x) ** 2
            + (new_y - old_y) ** 2
        )

        required_distance = new_radius + old_radius + minimum_gap

        if center_distance < required_distance:
            return True

    return False


def draw_one_particle(image, mask, center_x, center_y, radius, intensity):
    """
    Draw one nearly circular nanoparticle into the image and mask.

    image:
        Grayscale SEM-like image.
    mask:
        Binary mask of particle locations.
    center_x, center_y:
        Particle center coordinates.
    radius:
        Particle radius in pixels.
    intensity:
        Brightness of the particle.

    This function creates a filled disk and then brightens the particle
    boundary slightly so that the image looks more SEM-like.
    """

    height, width = image.shape

    y_indices, x_indices = np.indices((height, width))

    distance = np.sqrt(
        (x_indices - center_x) ** 2
        + (y_indices - center_y) ** 2
    )

    particle_region = distance <= radius

    image[particle_region] += intensity
    mask[particle_region] = 1.0

    edge_region = (
        (distance > 0.78 * radius)
        & (distance <= radius)
    )

    image[edge_region] += 0.12


def generate_one_sem_image(image_index):
    """
    Generate one synthetic SEM image and its segmentation mask.

    The particles are deliberately placed so that they do not overlap.

    Returns
    -------
    image : 2D NumPy array
        Grayscale image values between 0 and 1.
    mask : 2D NumPy array
        Binary particle mask.
    particle_table : pandas DataFrame
        One row per particle with radius and size category.
    """

    mode = choose_particle_size_mode(image_index)

    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

    image += np.random.normal(
        loc=0.32,
        scale=0.04,
        size=(IMAGE_SIZE, IMAGE_SIZE)
    ).astype(np.float32)

    if mode == "mostly_small":
        target_number_of_particles = np.random.randint(28, 44)
    elif mode == "mostly_medium":
        target_number_of_particles = np.random.randint(16, 28)
    elif mode == "mostly_large":
        target_number_of_particles = np.random.randint(8, 16)
    elif mode == "balanced":
        target_number_of_particles = np.random.randint(16, 30)
    elif mode == "small_large_mix":
        target_number_of_particles = np.random.randint(14, 28)
    else:
        target_number_of_particles = np.random.randint(16, 30)

    particle_records = []
    placed_particles = []

    for particle_number in range(target_number_of_particles):

        particle_was_placed = False

        for attempt in range(MAX_PLACEMENT_ATTEMPTS):

            radius = sample_particle_radius(mode)
            margin = int(math.ceil(radius)) + 2

            center_x = np.random.randint(margin, IMAGE_SIZE - margin)
            center_y = np.random.randint(margin, IMAGE_SIZE - margin)

            overlap = particles_overlap(
                new_x=center_x,
                new_y=center_y,
                new_radius=radius,
                placed_particles=placed_particles,
                minimum_gap=MINIMUM_PARTICLE_GAP,
            )

            if overlap:
                continue

            intensity = np.random.uniform(0.25, 0.55)

            draw_one_particle(
                image=image,
                mask=mask,
                center_x=center_x,
                center_y=center_y,
                radius=radius,
                intensity=intensity,
            )

            placed_particles.append(
                (center_x, center_y, radius)
            )

            particle_records.append(
                {
                    "Image_ID": f"SEM_{image_index:03d}",
                    "Particle_Number": len(placed_particles),
                    "Size_Mode": mode,
                    "Center_X_px": center_x,
                    "Center_Y_px": center_y,
                    "Radius_px": radius,
                    "Area_px": math.pi * radius * radius,
                    "Size_Class": classify_radius(radius),
                }
            )

            particle_was_placed = True
            break

        # If a particle could not be placed without overlap, simply skip it.
        if not particle_was_placed:
            continue

    image = ndi.gaussian_filter(image, sigma=1.0)

    image += np.random.normal(
        loc=0.0,
        scale=0.03,
        size=image.shape
    ).astype(np.float32)

    image = np.clip(image, 0.0, 1.0)

    particle_table = pd.DataFrame(particle_records)

    return image, mask, particle_table


def summarize_particle_table(particle_table):
    """
    Calculate per-image particle statistics from the true particle table.
    """

    small_count = int((particle_table["Size_Class"] == "small").sum())
    medium_count = int((particle_table["Size_Class"] == "medium").sum())
    large_count = int((particle_table["Size_Class"] == "large").sum())

    summary = {
        "Image_ID": particle_table["Image_ID"].iloc[0],
        "Size_Mode": particle_table["Size_Mode"].iloc[0],
        "True_Total_Particles": len(particle_table),
        "True_Small_Count": small_count,
        "True_Medium_Count": medium_count,
        "True_Large_Count": large_count,
        "True_Mean_Radius_px": particle_table["Radius_px"].mean(),
        "True_Mean_Area_px": particle_table["Area_px"].mean(),
    }

    return summary


def overlay_predicted_outlines(axis, binary_mask, color="lime", min_area=12):
    """
    Draw particle outlines for each connected component in a predicted mask.

    The outlines are drawn directly on a matplotlib axis that already shows
    the original SEM image.

    The mask is first labeled into connected particle regions. Very tiny
    regions are ignored because they are often just noise.
    """

    labeled_mask, number_of_regions = ndi.label(binary_mask > 0.5)

    for region_label in range(1, number_of_regions + 1):

        single_region = labeled_mask == region_label

        region_area = single_region.sum()

        if region_area < min_area:
            continue

        axis.contour(
            single_region.astype(float),
            levels=[0.5],
            colors=color,
            linewidths=1.2
        )


# ===========================================================================
# 3. GENERATE THE SYNTHETIC DATASET
# ===========================================================================

all_particle_tables = []
all_image_summaries = []

for image_index in range(NUMBER_OF_IMAGES):

    image_id = f"SEM_{image_index:03d}"

    image, mask, particle_table = generate_one_sem_image(image_index)

    plt.imsave(
        IMAGE_DIRECTORY / f"{image_id}.png",
        image,
        cmap="gray",
        vmin=0.0,
        vmax=1.0
    )

    plt.imsave(
        MASK_DIRECTORY / f"{image_id}_mask.png",
        mask,
        cmap="gray",
        vmin=0.0,
        vmax=1.0
    )

    particle_table.to_csv(
        DATASET_DIRECTORY / f"{image_id}_particles.csv",
        index=False
    )

    all_particle_tables.append(particle_table)
    all_image_summaries.append(summarize_particle_table(particle_table))

particle_metadata_df = pd.concat(
    all_particle_tables,
    ignore_index=True
)

image_summary_df = pd.DataFrame(all_image_summaries)

particle_metadata_df.to_csv(
    DATASET_DIRECTORY / "all_particle_metadata.csv",
    index=False
)

image_summary_df.to_csv(
    DATASET_DIRECTORY / "all_image_summaries.csv",
    index=False
)

print("Generated synthetic dataset.")
print("Number of images:", NUMBER_OF_IMAGES)
print("Total number of particles:", len(particle_metadata_df))
print()


# ---------------------------------------------------------------------------
# VERIFY THAT THE GENERATED IMAGE FILES CAN BE READ
# ---------------------------------------------------------------------------

# Before starting CNN training, test one generated PNG file.
#
# This gives a much clearer error message if there is a filesystem problem.

example_access_file = (
    IMAGE_DIRECTORY
    / "SEM_000.png"
)

try:

    with open(
        example_access_file,
        "rb"
    ) as file:

        file.read(1)

except PermissionError as error:

    raise PermissionError(
        "\nPython generated the SEM images but cannot read them back.\n"
        "This usually indicates a filesystem permission or synchronization "
        "problem.\n\n"
        f"Problem file:\n{example_access_file}\n\n"
        "The recommended course setup is to store generated CNN data in the "
        "local user home directory rather than a synced Windows/OneDrive "
        "directory."
    ) from error


print("Generated image read test: passed")
print()


# ===========================================================================
# 4. VISUALIZE THE SYNTHETIC DATASET
# ===========================================================================

example_indices = [0, 1, 2, 3, 4, 5]

figure, axes = plt.subplots(
    nrows=2,
    ncols=3,
    figsize=(10, 7)
)

for axis, example_index in zip(axes.flatten(), example_indices):

    image_id = f"SEM_{example_index:03d}"

    image = plt.imread(
        IMAGE_DIRECTORY / f"{image_id}.png"
    )

    axis.imshow(image, cmap="gray")
    axis.set_title(
        image_summary_df.loc[example_index, "Size_Mode"]
    )
    axis.axis("off")

figure.suptitle(
    "Example Synthetic SEM Images (Non-overlapping Particles)",
    fontsize=14
)

figure.tight_layout()

figure.savefig(
    RESULTS_DIRECTORY / "01_example_sem_images.png",
    dpi=200
)

plt.show()


# ===========================================================================
# 5. TRAIN/TEST SPLIT
# ===========================================================================

all_image_ids = [f"SEM_{image_index:03d}" for image_index in range(NUMBER_OF_IMAGES)]

random.shuffle(all_image_ids)

split_index = int(TRAIN_FRACTION * len(all_image_ids))

train_image_ids = all_image_ids[:split_index]
test_image_ids = all_image_ids[split_index:]

split_table = pd.DataFrame(
    {
        "Image_ID": train_image_ids + test_image_ids,
        "Split": ["train"] * len(train_image_ids) + ["test"] * len(test_image_ids),
    }
)

split_table.to_csv(
    DATASET_DIRECTORY / "train_test_split.csv",
    index=False
)

print("Training images:", len(train_image_ids))
print("Test images:", len(test_image_ids))
print()


# ===========================================================================
# 6. DATASET CLASS FOR PYTORCH
# ===========================================================================

class SEMDataset(Dataset):
    """
    PyTorch dataset for loading SEM images and masks.
    """

    def __init__(self, image_ids, image_directory, mask_directory):

        self.image_ids = image_ids
        self.image_directory = image_directory
        self.mask_directory = mask_directory

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):

        image_id = self.image_ids[index]

        image = plt.imread(
            self.image_directory / f"{image_id}.png"
        ).astype(np.float32)

        mask = plt.imread(
            self.mask_directory / f"{image_id}_mask.png"
        ).astype(np.float32)

        if image.ndim == 3:
            image = image[:, :, 0]

        if mask.ndim == 3:
            mask = mask[:, :, 0]

        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        image_tensor = torch.tensor(
            image,
            dtype=torch.float32
        )

        mask_tensor = torch.tensor(
            mask,
            dtype=torch.float32
        )

        return image_tensor, mask_tensor, image_id


train_dataset = SEMDataset(
    train_image_ids,
    IMAGE_DIRECTORY,
    MASK_DIRECTORY
)

test_dataset = SEMDataset(
    test_image_ids,
    IMAGE_DIRECTORY,
    MASK_DIRECTORY
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ===========================================================================
# 7. CNN MODEL
# ===========================================================================

class SimpleSegmentationCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=1),
        )

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x


model = SimpleSegmentationCNN().to(DEVICE)

print("CNN architecture:")
print(model)
print()


# ===========================================================================
# 8. LOSS FUNCTION AND OPTIMIZER
# ===========================================================================

loss_function = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ===========================================================================
# 9. TRAIN THE CNN
# ===========================================================================

training_history = []

for epoch in range(1, NUMBER_OF_EPOCHS + 1):

    model.train()

    epoch_loss = 0.0

    for batch_images, batch_masks, _ in train_loader:

        batch_images = batch_images.to(DEVICE)
        batch_masks = batch_masks.to(DEVICE)

        optimizer.zero_grad()

        predicted_logits = model(batch_images)

        loss = loss_function(
            predicted_logits,
            batch_masks
        )

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    average_epoch_loss = epoch_loss / len(train_loader)

    training_history.append(
        {
            "Epoch": epoch,
            "Training_Loss": average_epoch_loss,
        }
    )

    print(
        f"Epoch {epoch:3d} | "
        f"Training loss = {average_epoch_loss:.5f}"
    )

training_history_df = pd.DataFrame(training_history)

training_history_df.to_csv(
    RESULTS_DIRECTORY / "cnn_training_history.csv",
    index=False
)

print()


# ===========================================================================
# 10. PLOT THE TRAINING LOSS
# ===========================================================================

plt.figure(figsize=(6, 4))

plt.plot(
    training_history_df["Epoch"],
    training_history_df["Training_Loss"]
)

plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("CNN Training Loss")
plt.tight_layout()

plt.savefig(
    RESULTS_DIRECTORY / "02_cnn_training_loss.png",
    dpi=200
)

plt.show()


# ===========================================================================
# 11. SEGMENT THE TEST IMAGES
# ===========================================================================

def sigmoid_to_binary_mask(logits, threshold=0.5):
    """
    Convert raw CNN output into a binary segmentation mask.
    """

    probabilities = torch.sigmoid(logits)
    binary_mask = (probabilities >= threshold).float()

    return binary_mask


model.eval()

all_test_predictions = []

with torch.no_grad():

    for batch_images, batch_masks, batch_image_ids in test_loader:

        batch_images = batch_images.to(DEVICE)

        predicted_logits = model(batch_images)

        predicted_masks = sigmoid_to_binary_mask(predicted_logits)

        predicted_masks_numpy = predicted_masks.cpu().numpy()
        true_masks_numpy = batch_masks.numpy()
        images_numpy = batch_images.cpu().numpy()

        for image_number in range(len(batch_image_ids)):

            image_id = batch_image_ids[image_number]

            image_array = images_numpy[image_number, 0]
            true_mask_array = true_masks_numpy[image_number, 0]
            predicted_mask_array = predicted_masks_numpy[image_number, 0]

            np.save(
                RESULTS_DIRECTORY / f"{image_id}_predicted_mask.npy",
                predicted_mask_array
            )

            plt.imsave(
                RESULTS_DIRECTORY / f"{image_id}_predicted_mask.png",
                predicted_mask_array,
                cmap="gray",
                vmin=0.0,
                vmax=1.0
            )

            all_test_predictions.append(
                {
                    "Image_ID": image_id,
                    "Image_Array": image_array,
                    "True_Mask_Array": true_mask_array,
                    "Predicted_Mask_Array": predicted_mask_array,
                }
            )


# ===========================================================================
# 12. VISUALIZE SEGMENTATION RESULTS
# ===========================================================================

# Show several test images with:
#   1. The original image
#   2. The true mask
#   3. The predicted mask
#   4. The original image overlaid with CNN-predicted outlines

number_of_examples_to_show = 4

figure, axes = plt.subplots(
    nrows=number_of_examples_to_show,
    ncols=4,
    figsize=(12, 10)
)

for row_index in range(number_of_examples_to_show):

    example = all_test_predictions[row_index]

    axes[row_index, 0].imshow(
        example["Image_Array"],
        cmap="gray"
    )
    axes[row_index, 0].set_title(
        f'{example["Image_ID"]} Image'
    )
    axes[row_index, 0].axis("off")

    axes[row_index, 1].imshow(
        example["True_Mask_Array"],
        cmap="gray"
    )
    axes[row_index, 1].set_title("True Mask")
    axes[row_index, 1].axis("off")

    axes[row_index, 2].imshow(
        example["Predicted_Mask_Array"],
        cmap="gray"
    )
    axes[row_index, 2].set_title("Predicted Mask")
    axes[row_index, 2].axis("off")

    axes[row_index, 3].imshow(
        example["Image_Array"],
        cmap="gray"
    )
    overlay_predicted_outlines(
        axes[row_index, 3],
        example["Predicted_Mask_Array"],
        color="lime"
    )
    axes[row_index, 3].set_title("Image + Predicted Outlines")
    axes[row_index, 3].axis("off")

figure.suptitle(
    "Example CNN Segmentation Results and Predicted Particle Outlines",
    fontsize=14
)

figure.tight_layout()

figure.savefig(
    RESULTS_DIRECTORY / "03_example_segmentation_results.png",
    dpi=200
)

plt.show()


# ===========================================================================
# 13. PARTICLE STATISTICS FROM MASKS
# ===========================================================================

def particle_statistics_from_mask(mask, image_id):
    """
    Estimate per-image particle statistics from a binary mask.

    The function:
        1. Labels connected particle regions
        2. Computes each particle area
        3. Converts area into an equivalent circular radius
        4. Counts small, medium, and large particles
    """

    labeled_mask, number_of_particles = ndi.label(mask > 0.5)

    object_slices = ndi.find_objects(labeled_mask)

    equivalent_radii = []

    for label_index, object_slice in enumerate(object_slices, start=1):

        if object_slice is None:
            continue

        particle_region = labeled_mask[object_slice] == label_index
        area = particle_region.sum()

        if area < 12:
            continue

        equivalent_radius = math.sqrt(area / math.pi)
        equivalent_radii.append(equivalent_radius)

    small_count = 0
    medium_count = 0
    large_count = 0

    for radius in equivalent_radii:

        if radius < SMALL_RADIUS_MAX:
            small_count += 1
        elif radius < MEDIUM_RADIUS_MAX:
            medium_count += 1
        else:
            large_count += 1

    if len(equivalent_radii) == 0:
        mean_radius = 0.0
    else:
        mean_radius = float(np.mean(equivalent_radii))

    result = {
        "Image_ID": image_id,
        "Estimated_Total_Particles": len(equivalent_radii),
        "Estimated_Small_Count": small_count,
        "Estimated_Medium_Count": medium_count,
        "Estimated_Large_Count": large_count,
        "Estimated_Mean_Radius_px": mean_radius,
    }

    return result


true_statistics = []
predicted_statistics = []

for example in all_test_predictions:

    image_id = example["Image_ID"]

    true_stats = particle_statistics_from_mask(
        example["True_Mask_Array"],
        image_id
    )

    pred_stats = particle_statistics_from_mask(
        example["Predicted_Mask_Array"],
        image_id
    )

    true_statistics.append(true_stats)
    predicted_statistics.append(pred_stats)

true_statistics_df = pd.DataFrame(true_statistics)
predicted_statistics_df = pd.DataFrame(predicted_statistics)

comparison_df = true_statistics_df.merge(
    predicted_statistics_df,
    on="Image_ID",
    suffixes=("_From_True_Mask", "_From_Predicted_Mask")
)

comparison_df.to_csv(
    RESULTS_DIRECTORY / "test_image_particle_statistics.csv",
    index=False
)


# ===========================================================================
# 14. MERGE WITH THE TRUE GENERATION SUMMARY
# ===========================================================================

full_test_summary_df = comparison_df.merge(
    image_summary_df,
    on="Image_ID"
)

full_test_summary_df.to_csv(
    RESULTS_DIRECTORY / "test_image_statistics_with_generation_truth.csv",
    index=False
)


# ===========================================================================
# 15. SUMMARY PLOTS FOR PARTICLE STATISTICS
# ===========================================================================

figure, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(11, 5)
)

axes[0].scatter(
    full_test_summary_df["Estimated_Total_Particles_From_True_Mask"],
    full_test_summary_df["Estimated_Total_Particles_From_Predicted_Mask"]
)

minimum_count = min(
    full_test_summary_df["Estimated_Total_Particles_From_True_Mask"].min(),
    full_test_summary_df["Estimated_Total_Particles_From_Predicted_Mask"].min()
)

maximum_count = max(
    full_test_summary_df["Estimated_Total_Particles_From_True_Mask"].max(),
    full_test_summary_df["Estimated_Total_Particles_From_Predicted_Mask"].max()
)

axes[0].plot(
    [minimum_count, maximum_count],
    [minimum_count, maximum_count],
    linestyle="--"
)

axes[0].set_xlabel("True Particle Count")
axes[0].set_ylabel("Predicted Particle Count")
axes[0].set_title("Particle Count Per Image")


axes[1].scatter(
    full_test_summary_df["Estimated_Mean_Radius_px_From_True_Mask"],
    full_test_summary_df["Estimated_Mean_Radius_px_From_Predicted_Mask"]
)

minimum_radius = min(
    full_test_summary_df["Estimated_Mean_Radius_px_From_True_Mask"].min(),
    full_test_summary_df["Estimated_Mean_Radius_px_From_Predicted_Mask"].min()
)

maximum_radius = max(
    full_test_summary_df["Estimated_Mean_Radius_px_From_True_Mask"].max(),
    full_test_summary_df["Estimated_Mean_Radius_px_From_Predicted_Mask"].max()
)

axes[1].plot(
    [minimum_radius, maximum_radius],
    [minimum_radius, maximum_radius],
    linestyle="--"
)

axes[1].set_xlabel("True Mean Radius (px)")
axes[1].set_ylabel("Predicted Mean Radius (px)")
axes[1].set_title("Mean Particle Radius Per Image")

figure.suptitle(
    "CNN-Based Particle Statistics on Test Images",
    fontsize=14
)

figure.tight_layout()

figure.savefig(
    RESULTS_DIRECTORY / "04_particle_statistics_summary.png",
    dpi=200
)

plt.show()


# ===========================================================================
# 16. HISTOGRAM OF TRUE SIZE MODES
# ===========================================================================

mode_counts = image_summary_df["Size_Mode"].value_counts().sort_index()

plt.figure(figsize=(8, 4))

plt.bar(
    mode_counts.index,
    mode_counts.values
)

plt.xlabel("Image Mixture Mode")
plt.ylabel("Number of Images")
plt.title("Synthetic SEM Dataset Composition")
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig(
    RESULTS_DIRECTORY / "05_dataset_size_mode_distribution.png",
    dpi=200
)

plt.show()


# ===========================================================================
# 17. SAVE A SHORT SUMMARY TEXT FILE
# ===========================================================================

summary_file = RESULTS_DIRECTORY / "cnn_summary.txt"

with open(summary_file, "w") as file:

    file.write("CNN synthetic SEM nanoparticle segmentation\n")
    file.write("==========================================\n\n")

    file.write(f"Number of images: {NUMBER_OF_IMAGES}\n")
    file.write(f"Training images: {len(train_image_ids)}\n")
    file.write(f"Test images: {len(test_image_ids)}\n")
    file.write(f"Image size: {IMAGE_SIZE} x {IMAGE_SIZE}\n")
    file.write(f"Epochs: {NUMBER_OF_EPOCHS}\n")
    file.write(f"Batch size: {BATCH_SIZE}\n")
    file.write(f"Learning rate: {LEARNING_RATE}\n")
    file.write(f"Minimum particle gap: {MINIMUM_PARTICLE_GAP}\n\n")

    file.write("CNN model architecture:\n")
    file.write(str(model))
    file.write("\n")


# ===========================================================================
# 18. FINAL OUTPUT SUMMARY
# ===========================================================================

print("Analysis complete.")
print()
print("Main output files:")
print("01_example_sem_images.png")
print("02_cnn_training_loss.png")
print("03_example_segmentation_results.png")
print("04_particle_statistics_summary.png")
print("05_dataset_size_mode_distribution.png")
print("cnn_training_history.csv")
print("test_image_particle_statistics.csv")
print("test_image_statistics_with_generation_truth.csv")
print()


# ===========================================================================
# PRACTICE EXERCISES
# ===========================================================================

# 1. Change MINIMUM_PARTICLE_GAP from 2.0 to 4.0.
#    How does this change the appearance of the synthetic SEM images?
#
# 2. Change NUMBER_OF_IMAGES from 180 to 60.
#    How does this affect the training results?
#
# 3. Change IMAGE_SIZE from 128 to 96 or 160.
#    How does the segmentation difficulty change?
#
# 4. Change NUMBER_OF_EPOCHS from 18 to 30.
#    Does the CNN improve?
#
# 5. Add another high-level size mode.
#    For example, create a mode that is almost entirely medium particles.
#
# 6. Change the CNN architecture.
#    Add one more convolutional layer in the encoder.
#
# 7. Change the segmentation threshold in sigmoid_to_binary_mask().
#    Try 0.4 or 0.6 instead of 0.5.
#
# 8. Add a new plot comparing true and predicted large-particle counts.
