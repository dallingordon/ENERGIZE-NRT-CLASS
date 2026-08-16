"""


A simple PyTorch neural network for predicting melting temperature.

This example focuses on the neural-network training process:
    1. Read previously generated material data
    2. Build feature vectors
    3. Standardize the features
    4. Convert the data to PyTorch tensors
    5. Build a small neural network
    6. Perform forward propagation
    7. Calculate a loss
    8. Perform backpropagation
    9. Update the model weights
   10. Predict melting temperatures for training and test materials
   11. Save diagnostic plots and data

Expected folder layout
----------------------

course_examples/
|
|-- intro_to_data_science/
|   |
|   |-- structured_data/
|   |   |-- raw_material_files/
|   |       |-- MAT_001.csv
|   |       |-- MAT_002.csv
|   |       |-- ...
|   |
|   |-- basic_regression/
|       |-- regression_results/
|           |-- new_material_melting_temperature_predictions.csv
|
|-- ml_theory/
    |-- neural_networks/
        |-- 12_pytorch_neural_network_melting_temperature.py

The neural-network script therefore moves TWO directories upward before
entering the earlier intro_to_data_science directory.

The materials are synthetic and do not represent specific real materials.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ===========================================================================
# 1. PATHS
# ===========================================================================

# Directory containing this script.
SCRIPT_DIRECTORY = Path(__file__).resolve().parent

# Training data generated in the earlier structured-data example.
TRAINING_DATA_DIRECTORY = (
    SCRIPT_DIRECTORY
    / ".."
    / ".."
    / "intro_to_data_science"
    / "structured_data"
    / "raw_material_files"
).resolve()

# The 25 new materials generated in the earlier regression example.
TEST_DATA_FILE = (
    SCRIPT_DIRECTORY
    / ".."
    / ".."
    / "intro_to_data_science"
    / "basic_regression"
    / "regression_results"
    / "new_material_melting_temperature_predictions.csv"
).resolve()

# Results from this script will be saved beside the script.
RESULTS_DIRECTORY = SCRIPT_DIRECTORY / "nn_results"
RESULTS_DIRECTORY.mkdir(exist_ok=True)

print("=" * 70)
print("PyTorch Neural Network: Melting Temperature Prediction")
print("=" * 70)
print()
print("Training data:")
print(TRAINING_DATA_DIRECTORY)
print()
print("Test data:")
print(TEST_DATA_FILE)
print()
print("Results:")
print(RESULTS_DIRECTORY)
print()


# ===========================================================================
# 2. SETTINGS
# ===========================================================================

RANDOM_SEED = 42

# Number of complete passes through the training data.
NUMBER_OF_EPOCHS = 1200

# Controls the size of each weight update.
LEARNING_RATE = 0.01

# Print training progress every this many epochs.
PRINT_EVERY = 100

# This introductory example intentionally uses the CPU.
DEVICE = torch.device("cpu")

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

print("PyTorch device:", DEVICE)
print()


# ===========================================================================
# 3. FEATURES AND TARGET
# ===========================================================================

# These five properties form one feature vector for each material.
FEATURE_COLUMNS = [
    "Vacancy_Formation_Energy_eV",
    "Bulk_Modulus_GPa",
    "Average_Surface_Energy_J_m2",
    "Stacking_Fault_Energy_mJ_m2",
    "Average_Bond_Strength_eV",
]

# Melting temperature is NOT an input feature.
# It is the value that the network will learn to predict.
TARGET_COLUMN = "Melting_Temperature_K"

SHORT_FEATURE_NAMES = [
    "Vacancy E",
    "Bulk Modulus",
    "Surface E",
    "Stacking Fault E",
    "Bond Strength",
]


# ===========================================================================
# 4. READ THE 100 TRAINING MATERIALS
# ===========================================================================

training_files = sorted(
    TRAINING_DATA_DIRECTORY.glob("MAT_*.csv")
)

if len(training_files) == 0:
    raise FileNotFoundError(
        "\nNo training CSV files were found.\n"
        f"Python looked in:\n{TRAINING_DATA_DIRECTORY}\n"
    )

training_tables = []

for file_name in training_files:
    material_table = pd.read_csv(file_name)
    training_tables.append(material_table)

training_df = pd.concat(
    training_tables,
    ignore_index=True
)

print("Training materials found:", len(training_df))
print(training_df.head())
print()


# ===========================================================================
# 5. READ THE 25 TEST MATERIALS
# ===========================================================================

if not TEST_DATA_FILE.exists():
    raise FileNotFoundError(
        "\nThe test CSV file was not found.\n"
        f"Python looked for:\n{TEST_DATA_FILE}\n"
    )

test_df = pd.read_csv(TEST_DATA_FILE)

print("Test materials found:", len(test_df))
print()


# ===========================================================================
# 6. CREATE FEATURE MATRICES AND TARGET ARRAYS
# ===========================================================================

# X contains the input features.
# y contains the target melting temperatures.

X_train = training_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
y_train = training_df[TARGET_COLUMN].to_numpy(dtype=np.float32)

X_test = test_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
y_test = test_df[TARGET_COLUMN].to_numpy(dtype=np.float32)

print("Training feature matrix shape:", X_train.shape)
print("Training target shape:", y_train.shape)
print()
print("Example feature vector:")
print(X_train[0])
print("True melting temperature:", y_train[0], "K")
print()


# ===========================================================================
# 7. STANDARDIZE THE FEATURES
# ===========================================================================

# The features have very different numerical scales.
#
# For example:
#     vacancy formation energy may be around 1 to 4
#     bulk modulus may be around 50 to 300
#
# Neural networks usually train more easily when the input features are on
# similar scales.

feature_scaler = StandardScaler()

# fit_transform() learns the scaling ONLY from the training data.
X_train_scaled = feature_scaler.fit_transform(X_train).astype(np.float32)

# transform() uses the SAME scaling for the test data.
X_test_scaled = feature_scaler.transform(X_test).astype(np.float32)


# Save feature statistics before and after scaling.

before_scaling_df = pd.DataFrame({
    "Feature": FEATURE_COLUMNS,
    "Mean": X_train.mean(axis=0),
    "Standard_Deviation": X_train.std(axis=0),
    "Minimum": X_train.min(axis=0),
    "Maximum": X_train.max(axis=0),
})

after_scaling_df = pd.DataFrame({
    "Feature": FEATURE_COLUMNS,
    "Mean": X_train_scaled.mean(axis=0),
    "Standard_Deviation": X_train_scaled.std(axis=0),
    "Minimum": X_train_scaled.min(axis=0),
    "Maximum": X_train_scaled.max(axis=0),
})

before_scaling_df.to_csv(
    RESULTS_DIRECTORY / "feature_summary_before_scaling.csv",
    index=False
)

after_scaling_df.to_csv(
    RESULTS_DIRECTORY / "feature_summary_after_scaling.csv",
    index=False
)


# ===========================================================================
# 8. FEATURIZATION DIAGNOSTIC PLOT
# ===========================================================================

figure, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(14, 5)
)

axes[0].boxplot(
    X_train,
    tick_labels=SHORT_FEATURE_NAMES
)
axes[0].set_title("Before Scaling")
axes[0].set_ylabel("Raw Numerical Value")
axes[0].tick_params(axis="x", rotation=30)

axes[1].boxplot(
    X_train_scaled,
    tick_labels=SHORT_FEATURE_NAMES
)
axes[1].set_title("After Standardization")
axes[1].set_ylabel("Standardized Value")
axes[1].tick_params(axis="x", rotation=30)

figure.suptitle("Neural-Network Input Featurization")
figure.tight_layout()

figure.savefig(
    RESULTS_DIRECTORY / "01_feature_scaling_diagnostics.png",
    dpi=200
)

plt.show()


# ===========================================================================
# 9. FEATURE VERSUS MELTING-TEMPERATURE PLOT
# ===========================================================================

figure, axes = plt.subplots(
    nrows=2,
    ncols=3,
    figsize=(14, 8)
)

axes = axes.flatten()

for feature_index in range(len(FEATURE_COLUMNS)):

    axes[feature_index].scatter(
        X_train[:, feature_index],
        y_train
    )

    axes[feature_index].set_xlabel(
        SHORT_FEATURE_NAMES[feature_index]
    )

    axes[feature_index].set_ylabel(
        "Melting Temperature (K)"
    )

axes[5].axis("off")

figure.suptitle("Training Features Versus Melting Temperature")
figure.tight_layout()

figure.savefig(
    RESULTS_DIRECTORY / "02_feature_target_relationships.png",
    dpi=200
)

plt.show()


# ===========================================================================
# 10. CONVERT NUMPY ARRAYS TO PYTORCH TENSORS
# ===========================================================================

# PyTorch neural networks operate on tensors.

X_train_tensor = torch.tensor(
    X_train_scaled,
    dtype=torch.float32,
    device=DEVICE
)

y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.float32,
    device=DEVICE
).reshape(-1, 1)

X_test_tensor = torch.tensor(
    X_test_scaled,
    dtype=torch.float32,
    device=DEVICE
)

print("PyTorch training feature tensor:", X_train_tensor.shape)
print("PyTorch training target tensor:", y_train_tensor.shape)
print()


# ===========================================================================
# 11. BUILD THE NEURAL NETWORK
# ===========================================================================

# Architecture:
#
#   5 input features
#         |
#         v
#   16 hidden neurons
#         |
#       ReLU
#         |
#         v
#    8 hidden neurons
#         |
#       ReLU
#         |
#         v
#    1 predicted melting temperature

class MeltingTemperatureNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.layer_1 = nn.Linear(5, 16)
        self.activation_1 = nn.ReLU()

        self.layer_2 = nn.Linear(16, 8)
        self.activation_2 = nn.ReLU()

        self.output_layer = nn.Linear(8, 1)

    def forward(self, x):

        x = self.layer_1(x)
        x = self.activation_1(x)

        x = self.layer_2(x)
        x = self.activation_2(x)

        prediction = self.output_layer(x)

        return prediction


model = MeltingTemperatureNN().to(DEVICE)

print("Neural-network architecture:")
print(model)
print()

number_of_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print("Trainable parameters:", number_of_parameters)
print()


# ===========================================================================
# 12. LOSS FUNCTION AND OPTIMIZER
# ===========================================================================

# Mean squared error tells the network how wrong its predictions are.
loss_function = nn.MSELoss()

# Adam uses the gradients to update the model parameters.
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ===========================================================================
# 13. LOOK AT PREDICTIONS BEFORE TRAINING
# ===========================================================================

model.eval()

with torch.no_grad():

    initial_predictions = model(
        X_train_tensor[:5]
    ).squeeze().cpu().numpy()

print("Predictions before training:")
print(initial_predictions)
print()
print("True values:")
print(y_train[:5])
print()


# ===========================================================================
# 14. TRAIN THE NEURAL NETWORK
# ===========================================================================

# We use full-batch training here.
#
# Every epoch uses all 100 training materials at once. This keeps the example
# simple and makes the backpropagation diagnostics easier to interpret.

history = []

for epoch in range(1, NUMBER_OF_EPOCHS + 1):

    model.train()

    # -----------------------------------------------------------------------
    # A. Clear gradients left over from the previous epoch.
    # -----------------------------------------------------------------------

    optimizer.zero_grad()

    # -----------------------------------------------------------------------
    # B. FORWARD PROPAGATION
    #
    # Pass the input feature vectors through the neural network.
    # -----------------------------------------------------------------------

    predictions = model(X_train_tensor)

    # -----------------------------------------------------------------------
    # C. CALCULATE THE LOSS
    #
    # Compare predicted and true melting temperatures.
    # -----------------------------------------------------------------------

    loss = loss_function(
        predictions,
        y_train_tensor
    )

    # -----------------------------------------------------------------------
    # D. BACKPROPAGATION
    #
    # Calculate the gradient of the loss with respect to every trainable
    # weight and bias in the network.
    # -----------------------------------------------------------------------

    loss.backward()

    # -----------------------------------------------------------------------
    # E. RECORD GRADIENT MAGNITUDES
    #
    # These values are recorded AFTER backward() and BEFORE optimizer.step().
    # -----------------------------------------------------------------------

    gradient_layer_1 = model.layer_1.weight.grad.norm().item()
    gradient_layer_2 = model.layer_2.weight.grad.norm().item()
    gradient_output = model.output_layer.weight.grad.norm().item()

    total_gradient_squared = 0.0

    for parameter in model.parameters():

        if parameter.grad is not None:

            gradient_norm = parameter.grad.norm().item()

            total_gradient_squared += gradient_norm ** 2

    total_gradient_norm = total_gradient_squared ** 0.5

    # -----------------------------------------------------------------------
    # F. UPDATE THE WEIGHTS
    #
    # This is the step where the network actually changes its parameters.
    # -----------------------------------------------------------------------

    optimizer.step()

    # -----------------------------------------------------------------------
    # G. SAVE TRAINING DIAGNOSTICS
    # -----------------------------------------------------------------------

    prediction_array = (
        predictions
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    training_mae = mean_absolute_error(
        y_train,
        prediction_array
    )

    training_rmse = np.sqrt(
        mean_squared_error(
            y_train,
            prediction_array
        )
    )

    training_r2 = r2_score(
        y_train,
        prediction_array
    )

    history.append({
        "Epoch": epoch,
        "MSE_Loss": loss.item(),
        "RMSE_K": training_rmse,
        "MAE_K": training_mae,
        "R2": training_r2,
        "Layer_1_Gradient_Norm": gradient_layer_1,
        "Layer_2_Gradient_Norm": gradient_layer_2,
        "Output_Layer_Gradient_Norm": gradient_output,
        "Total_Gradient_Norm": total_gradient_norm,
    })

    if (
        epoch == 1
        or epoch % PRINT_EVERY == 0
        or epoch == NUMBER_OF_EPOCHS
    ):

        print(
            f"Epoch {epoch:4d} | "
            f"MSE = {loss.item():10.2f} | "
            f"MAE = {training_mae:7.2f} K | "
            f"R2 = {training_r2:6.3f} | "
            f"Gradient norm = {total_gradient_norm:10.2f}"
        )

print()


# ===========================================================================
# 15. SAVE TRAINING HISTORY
# ===========================================================================

history_df = pd.DataFrame(history)

history_df.to_csv(
    RESULTS_DIRECTORY / "nn_training_history.csv",
    index=False
)


# ===========================================================================
# 16. TRAINING AND BACKPROPAGATION DIAGNOSTICS
# ===========================================================================

# Four panels make the training process easier to inspect:
#
#   1. Loss
#   2. MAE
#   3. Total gradient magnitude
#   4. Gradient magnitude in each layer

figure, axes = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(13, 9)
)

axes[0, 0].plot(
    history_df["Epoch"],
    history_df["MSE_Loss"]
)
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("MSE Loss")
axes[0, 0].set_title("Training Loss")

axes[0, 1].plot(
    history_df["Epoch"],
    history_df["MAE_K"]
)
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("MAE (K)")
axes[0, 1].set_title("Training Error")

axes[1, 0].plot(
    history_df["Epoch"],
    history_df["Total_Gradient_Norm"]
)
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("Total Gradient Norm")
axes[1, 0].set_title("Backpropagation: Total Gradient")

axes[1, 1].plot(
    history_df["Epoch"],
    history_df["Layer_1_Gradient_Norm"],
    label="Layer 1"
)
axes[1, 1].plot(
    history_df["Epoch"],
    history_df["Layer_2_Gradient_Norm"],
    label="Layer 2"
)
axes[1, 1].plot(
    history_df["Epoch"],
    history_df["Output_Layer_Gradient_Norm"],
    label="Output Layer"
)
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("Weight Gradient Norm")
axes[1, 1].set_title("Backpropagation: Gradient by Layer")
axes[1, 1].legend()

figure.suptitle("Neural-Network Training and Backpropagation")
figure.tight_layout()

figure.savefig(
    RESULTS_DIRECTORY / "03_training_and_backpropagation.png",
    dpi=200
)

plt.show()


# ===========================================================================
# 17. MAKE FINAL PREDICTIONS
# ===========================================================================

model.eval()

with torch.no_grad():

    train_predictions = (
        model(X_train_tensor)
        .cpu()
        .numpy()
        .reshape(-1)
    )

    test_predictions = (
        model(X_test_tensor)
        .cpu()
        .numpy()
        .reshape(-1)
    )


# ===========================================================================
# 18. CALCULATE FINAL METRICS
# ===========================================================================

train_mae = mean_absolute_error(
    y_train,
    train_predictions
)

train_rmse = np.sqrt(
    mean_squared_error(
        y_train,
        train_predictions
    )
)

train_r2 = r2_score(
    y_train,
    train_predictions
)

test_mae = mean_absolute_error(
    y_test,
    test_predictions
)

test_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        test_predictions
    )
)

test_r2 = r2_score(
    y_test,
    test_predictions
)

print("=" * 70)
print("Final model performance")
print("=" * 70)

print(
    f"Training: MAE = {train_mae:.1f} K, "
    f"RMSE = {train_rmse:.1f} K, "
    f"R2 = {train_r2:.3f}"
)

print(
    f"Test:     MAE = {test_mae:.1f} K, "
    f"RMSE = {test_rmse:.1f} K, "
    f"R2 = {test_r2:.3f}"
)

print()


# ===========================================================================
# 19. SAVE PREDICTIONS
# ===========================================================================

training_predictions_df = pd.DataFrame({
    "Material_ID": training_df["Material_ID"],
    "True_Melting_Temperature_K": y_train,
    "NN_Predicted_Melting_Temperature_K": train_predictions,
    "Absolute_Error_K": np.abs(train_predictions - y_train),
})

test_predictions_df = pd.DataFrame({
    "Material_ID": test_df["Material_ID"],
    "True_Melting_Temperature_K": y_test,
    "NN_Predicted_Melting_Temperature_K": test_predictions,
    "Absolute_Error_K": np.abs(test_predictions - y_test),
})

training_predictions_df.round(3).to_csv(
    RESULTS_DIRECTORY / "nn_training_predictions.csv",
    index=False
)

test_predictions_df.round(3).to_csv(
    RESULTS_DIRECTORY / "nn_test_predictions.csv",
    index=False
)


# ===========================================================================
# 20. TRAINING AND TEST PARITY PLOTS
# ===========================================================================

all_values = np.concatenate([
    y_train,
    train_predictions,
    y_test,
    test_predictions,
])

plot_minimum = np.min(all_values) - 100
plot_maximum = np.max(all_values) + 100

figure, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(12, 5)
)

# Training set
axes[0].scatter(
    y_train,
    train_predictions
)
axes[0].plot(
    [plot_minimum, plot_maximum],
    [plot_minimum, plot_maximum],
    linestyle="--"
)
axes[0].set_xlim(plot_minimum, plot_maximum)
axes[0].set_ylim(plot_minimum, plot_maximum)
axes[0].set_xlabel("True Melting Temperature (K)")
axes[0].set_ylabel("Predicted Melting Temperature (K)")
axes[0].set_title(
    "Training Set"
    + f"\nR2 = {train_r2:.3f}, MAE = {train_mae:.1f} K"
)

# Test set
axes[1].scatter(
    y_test,
    test_predictions
)
axes[1].plot(
    [plot_minimum, plot_maximum],
    [plot_minimum, plot_maximum],
    linestyle="--"
)
axes[1].set_xlim(plot_minimum, plot_maximum)
axes[1].set_ylim(plot_minimum, plot_maximum)
axes[1].set_xlabel("True Melting Temperature (K)")
axes[1].set_ylabel("Predicted Melting Temperature (K)")
axes[1].set_title(
    "25 New Test Materials"
    + f"\nR2 = {test_r2:.3f}, MAE = {test_mae:.1f} K"
)

figure.suptitle("Neural-Network Melting Temperature Prediction")
figure.tight_layout()

figure.savefig(
    RESULTS_DIRECTORY / "04_nn_melting_temperature_parity.png",
    dpi=200
)

plt.show()


# ===========================================================================
# 21. FINAL SUMMARY
# ===========================================================================

print("Analysis complete.")
print()
print("Results saved in:")
print(RESULTS_DIRECTORY)
print()

print("Main output files:")
print("01_feature_scaling_diagnostics.png")
print("02_feature_target_relationships.png")
print("03_training_and_backpropagation.png")
print("04_nn_melting_temperature_parity.png")
print("nn_training_history.csv")
print("nn_training_predictions.csv")
print("nn_test_predictions.csv")
print()


# ===========================================================================
# PRACTICE EXERCISES
# ===========================================================================

# 1. Change layer_1 from 16 neurons to 8 neurons.
#    How does performance change?
#
# 2. Change layer_2 from 8 neurons to 4 neurons.
#    Does the smaller network perform differently?
#
# 3. Change LEARNING_RATE from 0.01 to 0.001.
#    What happens to the loss curve?
#
# 4. Change LEARNING_RATE from 0.01 to 0.1.
#    What happens to the gradient curves?
#
# 5. Replace nn.ReLU() with nn.Tanh().
#    How does this affect training and backpropagation?
#
# 6. Compare the gradient magnitudes in the three layers.
#    Are they always the same?
#
# 7. Compare this neural network with the earlier linear regression,
#    polynomial regression, and kernel ridge regression models.
#    Is a more complicated model automatically better?
