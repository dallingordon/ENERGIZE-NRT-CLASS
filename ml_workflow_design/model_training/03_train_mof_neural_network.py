"""
STAGE 3: MODEL TRAINING
=======================

Train a basic PyTorch neural network to predict synthetic CO2 uptake.

This script does NOT regenerate data and does NOT redo preprocessing. It uses
exactly the files written by Stage 2. This separation is a major part of the
workflow-design lesson.
"""

from pathlib import Path
import json
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, r2_score


# ---------------------------------------------------------------------------
# 1. PATHS
# ---------------------------------------------------------------------------

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
PROJECT_DIRECTORY = SCRIPT_DIRECTORY.parent
PROCESSED_DIRECTORY = PROJECT_DIRECTORY / "workflow_data" / "processed"
MODEL_DIRECTORY = PROJECT_DIRECTORY / "workflow_data" / "model"
MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = PROCESSED_DIRECTORY / "train_processed.csv"
VALIDATION_FILE = PROCESSED_DIRECTORY / "validation_processed.csv"
PROCESSING_METADATA_FILE = PROCESSED_DIRECTORY / "processing_metadata.json"


# ---------------------------------------------------------------------------
# 2. MODEL-TRAINING CHOICES
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
NUMBER_OF_EPOCHS = 250
BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0

HIDDEN_LAYER_1 = 32
HIDDEN_LAYER_2 = 16

USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 30

DEVICE = torch.device("cpu")

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# 3. LOAD THE ARTIFACTS FROM STAGE 2
# ---------------------------------------------------------------------------

for file_name in [TRAIN_FILE, VALIDATION_FILE, PROCESSING_METADATA_FILE]:
    if not file_name.exists():
        raise FileNotFoundError("Run Stage 2 before Stage 3.")

train_df = pd.read_csv(TRAIN_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)

with open(PROCESSING_METADATA_FILE, "r") as file:
    processing_metadata = json.load(file)

feature_columns = processing_metadata["PROCESSED_FEATURE_COLUMNS"]
target_column = processing_metadata["TARGET_COLUMN"]

print("=" * 75)
print("STAGE 3: MODEL TRAINING")
print("=" * 75)
print("Features received from Stage 2:")
print(feature_columns)
print()


# ---------------------------------------------------------------------------
# 4. CONVERT TO TENSORS
# ---------------------------------------------------------------------------

X_train = train_df[feature_columns].to_numpy(dtype=np.float32)
y_train = train_df[target_column].to_numpy(dtype=np.float32).reshape(-1, 1)

X_validation = validation_df[feature_columns].to_numpy(dtype=np.float32)
y_validation = validation_df[target_column].to_numpy(dtype=np.float32).reshape(-1, 1)

train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

X_validation_tensor = torch.tensor(X_validation, dtype=torch.float32, device=DEVICE)
y_validation_tensor = torch.tensor(y_validation, dtype=torch.float32, device=DEVICE)


# ---------------------------------------------------------------------------
# 5. NEURAL NETWORK
# ---------------------------------------------------------------------------

class MOFAdsorptionNN(nn.Module):
    def __init__(self, number_of_features, hidden_1, hidden_2):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(number_of_features, hidden_1),
            nn.ReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, 1),
        )

    def forward(self, x):
        return self.network(x)


model = MOFAdsorptionNN(
    number_of_features=len(feature_columns),
    hidden_1=HIDDEN_LAYER_1,
    hidden_2=HIDDEN_LAYER_2,
).to(DEVICE)

print(model)
print()

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ---------------------------------------------------------------------------
# 6. TRAINING LOOP
# ---------------------------------------------------------------------------

history = []
best_validation_loss = np.inf
best_model_state = None
epochs_without_improvement = 0

for epoch in range(1, NUMBER_OF_EPOCHS + 1):

    model.train()
    batch_losses = []

    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = loss_function(predictions, batch_y)
        loss.backward()
        optimizer.step()

        batch_losses.append(loss.item())

    average_training_loss = float(np.mean(batch_losses))

    # Validation is separate from training.
    model.eval()

    with torch.no_grad():
        validation_predictions = model(X_validation_tensor)
        validation_loss = loss_function(
            validation_predictions,
            y_validation_tensor,
        ).item()

    validation_predictions_np = validation_predictions.cpu().numpy().reshape(-1)
    validation_true_np = y_validation.reshape(-1)

    validation_mae = mean_absolute_error(
        validation_true_np,
        validation_predictions_np,
    )

    validation_r2 = r2_score(
        validation_true_np,
        validation_predictions_np,
    )

    history.append(
        {
            "Epoch": epoch,
            "Training_MSE": average_training_loss,
            "Validation_MSE": validation_loss,
            "Validation_MAE": validation_mae,
            "Validation_R2": validation_r2,
        }
    )

    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        best_model_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epoch == 1 or epoch % 25 == 0:
        print(
            f"Epoch {epoch:3d} | "
            f"Train MSE = {average_training_loss:8.4f} | "
            f"Validation MSE = {validation_loss:8.4f} | "
            f"Validation R2 = {validation_r2:6.3f}"
        )

    if USE_EARLY_STOPPING and epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
        print("Early stopping at epoch", epoch)
        break

if best_model_state is not None:
    model.load_state_dict(best_model_state)


# ---------------------------------------------------------------------------
# 7. SAVE MODEL AND TRAINING HISTORY
# ---------------------------------------------------------------------------

history_df = pd.DataFrame(history)
history_df.to_csv(MODEL_DIRECTORY / "training_history.csv", index=False)

model_file = MODEL_DIRECTORY / "mof_adsorption_nn.pt"
torch.save(model.state_dict(), model_file)

model_metadata = {
    "FEATURE_COLUMNS": feature_columns,
    "TARGET_COLUMN": target_column,
    "NUMBER_OF_INPUT_FEATURES": len(feature_columns),
    "HIDDEN_LAYER_1": HIDDEN_LAYER_1,
    "HIDDEN_LAYER_2": HIDDEN_LAYER_2,
    "LEARNING_RATE": LEARNING_RATE,
    "BATCH_SIZE": BATCH_SIZE,
    "BEST_VALIDATION_MSE": float(best_validation_loss),
}

with open(MODEL_DIRECTORY / "model_metadata.json", "w") as file:
    json.dump(model_metadata, file, indent=4)


# ---------------------------------------------------------------------------
# 8. TRAINING DIAGNOSTICS
# ---------------------------------------------------------------------------

figure, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(history_df["Epoch"], history_df["Training_MSE"], label="Training")
axes[0].plot(history_df["Epoch"], history_df["Validation_MSE"], label="Validation")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE")
axes[0].set_title("Training and Validation Loss")
axes[0].legend()

axes[1].plot(history_df["Epoch"], history_df["Validation_R2"])
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Validation R2")
axes[1].set_title("Validation Performance")

figure.suptitle("Stage 3: Neural-Network Training", fontsize=15)
figure.tight_layout()
figure.savefig(MODEL_DIRECTORY / "03_training_diagnostics.png", dpi=200)
plt.show()


# ---------------------------------------------------------------------------
# 9. SUMMARY
# ---------------------------------------------------------------------------

print()
print("=" * 75)
print("STAGE 3 COMPLETE: MODEL TRAINING")
print("=" * 75)
print("Best validation MSE:", best_validation_loss)
print("Saved model:", model_file)
print()
print("Next: run 04_model_testing/04_test_mof_neural_network.py")
