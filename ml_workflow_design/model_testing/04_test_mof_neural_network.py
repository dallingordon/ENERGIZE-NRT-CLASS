"""
STAGE 4: MODEL TESTING
======================

Evaluate the trained neural network on data not used for model fitting.

This final stage asks whether the complete workflow generalizes. It also
checks whether prediction errors depend on pressure.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------------------------
# 1. PATHS
# ---------------------------------------------------------------------------

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
PROJECT_DIRECTORY = SCRIPT_DIRECTORY.parent
PROCESSED_DIRECTORY = PROJECT_DIRECTORY / "workflow_data" / "processed"
MODEL_DIRECTORY = PROJECT_DIRECTORY / "workflow_data" / "model"
TEST_RESULTS_DIRECTORY = PROJECT_DIRECTORY / "workflow_data" / "test_results"
TEST_RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = PROCESSED_DIRECTORY / "train_processed.csv"
TEST_FILE = PROCESSED_DIRECTORY / "test_processed.csv"
MODEL_FILE = MODEL_DIRECTORY / "mof_adsorption_nn.pt"
MODEL_METADATA_FILE = MODEL_DIRECTORY / "model_metadata.json"
PROCESSING_METADATA_FILE = PROCESSED_DIRECTORY / "processing_metadata.json"

DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# 2. LOAD WORKFLOW ARTIFACTS
# ---------------------------------------------------------------------------

for required_file in [
    TRAIN_FILE,
    TEST_FILE,
    MODEL_FILE,
    MODEL_METADATA_FILE,
    PROCESSING_METADATA_FILE,
]:
    if not required_file.exists():
        raise FileNotFoundError(
            f"Missing workflow artifact: {required_file}\n"
            "Run the earlier stages first."
        )

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

with open(MODEL_METADATA_FILE, "r") as file:
    model_metadata = json.load(file)

with open(PROCESSING_METADATA_FILE, "r") as file:
    processing_metadata = json.load(file)

feature_columns = model_metadata["FEATURE_COLUMNS"]
target_column = model_metadata["TARGET_COLUMN"]


# ---------------------------------------------------------------------------
# 3. REBUILD THE SAME NETWORK ARCHITECTURE
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
    number_of_features=model_metadata["NUMBER_OF_INPUT_FEATURES"],
    hidden_1=model_metadata["HIDDEN_LAYER_1"],
    hidden_2=model_metadata["HIDDEN_LAYER_2"],
).to(DEVICE)

model.load_state_dict(
    torch.load(MODEL_FILE, map_location=DEVICE)
)

model.eval()


# ---------------------------------------------------------------------------
# 4. PREDICTION FUNCTION
# ---------------------------------------------------------------------------

def predict_dataframe(dataframe):
    X = dataframe[feature_columns].to_numpy(dtype=np.float32)
    y_true = dataframe[target_column].to_numpy(dtype=np.float32)

    X_tensor = torch.tensor(X, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        y_pred = model(X_tensor).cpu().numpy().reshape(-1)

    return y_true, y_pred


# ---------------------------------------------------------------------------
# 5. PREDICT TRAINING AND TEST SETS
# ---------------------------------------------------------------------------

y_train, train_predictions = predict_dataframe(train_df)
y_test, test_predictions = predict_dataframe(test_df)


# ---------------------------------------------------------------------------
# 6. METRICS
# ---------------------------------------------------------------------------

def calculate_metrics(true_values, predicted_values):
    mae = mean_absolute_error(true_values, predicted_values)
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    r2 = r2_score(true_values, predicted_values)
    return mae, rmse, r2


train_mae, train_rmse, train_r2 = calculate_metrics(
    y_train,
    train_predictions,
)

test_mae, test_rmse, test_r2 = calculate_metrics(
    y_test,
    test_predictions,
)

print("=" * 75)
print("STAGE 4: MODEL TESTING")
print("=" * 75)
print()
print(f"Training MAE  = {train_mae:.3f} mmol/g")
print(f"Training RMSE = {train_rmse:.3f} mmol/g")
print(f"Training R2   = {train_r2:.3f}")
print()
print(f"Test MAE      = {test_mae:.3f} mmol/g")
print(f"Test RMSE     = {test_rmse:.3f} mmol/g")
print(f"Test R2       = {test_r2:.3f}")
print()


# ---------------------------------------------------------------------------
# 7. SAVE PREDICTIONS
# ---------------------------------------------------------------------------

training_results = pd.DataFrame(
    {
        "Sample_ID": train_df["Sample_ID"],
        "True_CO2_Uptake_mmol_g": y_train,
        "Predicted_CO2_Uptake_mmol_g": train_predictions,
        "Absolute_Error_mmol_g": np.abs(train_predictions - y_train),
    }
)

# Pressure_bar is intentionally preserved by Stage 2 for diagnostics.
test_results = pd.DataFrame(
    {
        "Sample_ID": test_df["Sample_ID"],
        "Pressure_bar": test_df["Pressure_bar"],
        "True_CO2_Uptake_mmol_g": y_test,
        "Predicted_CO2_Uptake_mmol_g": test_predictions,
        "Absolute_Error_mmol_g": np.abs(test_predictions - y_test),
    }
)

training_results.to_csv(
    TEST_RESULTS_DIRECTORY / "training_predictions.csv",
    index=False,
)

test_results.to_csv(
    TEST_RESULTS_DIRECTORY / "test_predictions.csv",
    index=False,
)


# ---------------------------------------------------------------------------
# 8. TRAINING AND TEST PARITY PLOTS
# ---------------------------------------------------------------------------

all_values = np.concatenate(
    [y_train, train_predictions, y_test, test_predictions]
)

plot_min = min(0.0, float(np.min(all_values)))
plot_max = float(np.max(all_values)) * 1.05

figure, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(y_train, train_predictions, s=18)
axes[0].plot([plot_min, plot_max], [plot_min, plot_max], linestyle="--")
axes[0].set_xlim(plot_min, plot_max)
axes[0].set_ylim(plot_min, plot_max)
axes[0].set_xlabel("True CO2 Uptake (mmol/g)")
axes[0].set_ylabel("Predicted CO2 Uptake (mmol/g)")
axes[0].set_title(
    "Training Set"
    + f"\nR2 = {train_r2:.3f}, MAE = {train_mae:.3f}"
)

axes[1].scatter(y_test, test_predictions, s=18)
axes[1].plot([plot_min, plot_max], [plot_min, plot_max], linestyle="--")
axes[1].set_xlim(plot_min, plot_max)
axes[1].set_ylim(plot_min, plot_max)
axes[1].set_xlabel("True CO2 Uptake (mmol/g)")
axes[1].set_ylabel("Predicted CO2 Uptake (mmol/g)")
axes[1].set_title(
    "Test Set"
    + f"\nR2 = {test_r2:.3f}, MAE = {test_mae:.3f}"
)

figure.suptitle("Stage 4: Final Model Parity", fontsize=15)
figure.tight_layout()
figure.savefig(
    TEST_RESULTS_DIRECTORY / "04_training_and_test_parity.png",
    dpi=200,
)
plt.show()


# ---------------------------------------------------------------------------
# 9. ERROR VERSUS PRESSURE
# ---------------------------------------------------------------------------

plt.figure(figsize=(7, 5))
plt.scatter(
    test_results["Pressure_bar"],
    test_results["Absolute_Error_mmol_g"],
    s=20,
)
plt.xscale("log")
plt.xlabel("Pressure (bar)")
plt.ylabel("Absolute Prediction Error (mmol/g)")
plt.title("Where Does the Model Make Errors?")
plt.tight_layout()
plt.savefig(
    TEST_RESULTS_DIRECTORY / "04_test_error_vs_pressure.png",
    dpi=200,
)
plt.show()


# ---------------------------------------------------------------------------
# 10. SAVE FINAL SUMMARY
# ---------------------------------------------------------------------------

summary = {
    "Training_MAE_mmol_g": float(train_mae),
    "Training_RMSE_mmol_g": float(train_rmse),
    "Training_R2": float(train_r2),
    "Test_MAE_mmol_g": float(test_mae),
    "Test_RMSE_mmol_g": float(test_rmse),
    "Test_R2": float(test_r2),
    "TEST_STRATEGY": processing_metadata["TEST_STRATEGY"],
    "LOG_TRANSFORM_PRESSURE": processing_metadata["LOG_TRANSFORM_PRESSURE"],
    "NORMALIZE_FEATURES": processing_metadata["NORMALIZE_FEATURES"],
    "FIT_PREPROCESSING_ON_TRAINING_ONLY": processing_metadata[
        "FIT_PREPROCESSING_ON_TRAINING_ONLY"
    ],
}

with open(TEST_RESULTS_DIRECTORY / "final_test_summary.json", "w") as file:
    json.dump(summary, file, indent=4)


# ---------------------------------------------------------------------------
# 11. INTERPRETATION PROMPTS
# ---------------------------------------------------------------------------

print("Workflow settings:")
print("  Log pressure:", processing_metadata["LOG_TRANSFORM_PRESSURE"])
print("  Normalize features:", processing_metadata["NORMALIZE_FEATURES"])
print("  Test strategy:", processing_metadata["TEST_STRATEGY"])
print()
print("Questions to ask:")
print("1. Is test performance close to training performance?")
print("2. Does prediction error depend on pressure?")
print("3. What changes when feature normalization is disabled?")
print("4. What changes when the pressure log transform is disabled?")
print("5. What happens with a high-pressure extrapolation test?")
print()
print("=" * 75)
print("WORKFLOW COMPLETE")
print("=" * 75)
