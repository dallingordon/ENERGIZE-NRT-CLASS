"""
STAGE 2: DATA PROCESSING
========================

Read the raw MOF adsorption data and prepare it for machine learning.

This stage is deliberately separated from model training. The neural network
should consume processed data rather than silently making preprocessing
choices inside the training script.

Important decisions near the top of this file include:

- whether pressure is log transformed
- whether features are normalized
- how missing values are handled
- how the test set is selected
- whether preprocessing statistics are fit only on training data
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# 1. PATHS
# ---------------------------------------------------------------------------

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
PROJECT_DIRECTORY = SCRIPT_DIRECTORY.parent

RAW_FILE = PROJECT_DIRECTORY / "workflow_data" / "raw" / "synthetic_mof_adsorption_raw.csv"
PROCESSED_DIRECTORY = PROJECT_DIRECTORY / "workflow_data" / "processed"
PROCESSED_DIRECTORY.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 2. PROCESSING CHOICES
# ---------------------------------------------------------------------------

RANDOM_SEED = 42

# Pressure spans ~0.001 to 50 bar. Try False and compare the model.
LOG_TRANSFORM_PRESSURE = False

# This is one of the most important switches in the exercise.
# Try False and rerun Stages 2, 3, and 4.
NORMALIZE_FEATURES = False

# Options: "median" or "drop"
MISSING_VALUE_METHOD = "median"

# Options: "random" or "high_pressure_extrapolation"
TEST_STRATEGY = "random"

TEST_FRACTION = 0.20
VALIDATION_FRACTION_OF_REMAINDER = 0.15
EXTRAPOLATION_PRESSURE_THRESHOLD_BAR = 15.0

# Only remove intentionally marked bad measurements from training.
REMOVE_FLAGGED_OUTLIERS_FROM_TRAINING = True

# Correct practice is True. False intentionally demonstrates data leakage.
FIT_PREPROCESSING_ON_TRAINING_ONLY = True


# ---------------------------------------------------------------------------
# 3. COLUMNS
# ---------------------------------------------------------------------------

RAW_FEATURE_COLUMNS = [
    "Surface_Area_m2_g",
    "Pore_Volume_cm3_g",
    "Average_Pore_Diameter_A",
    "Framework_Density_g_cm3",
    "Adsorption_Energy_kJ_mol",
    "Functional_Group_Polarity",
    "Defect_Fraction",
    "Temperature_K",
    "Pressure_bar",
]

TARGET_COLUMN = "CO2_Uptake_mmol_g"


# ---------------------------------------------------------------------------
# 4. READ RAW DATA
# ---------------------------------------------------------------------------

if not RAW_FILE.exists():
    raise FileNotFoundError(
        "Raw data were not found. Run Stage 1 before running Stage 2."
    )

raw_df = pd.read_csv(RAW_FILE)

print("=" * 75)
print("STAGE 2: DATA PROCESSING")
print("=" * 75)
print("Raw shape:", raw_df.shape)
print()


# ---------------------------------------------------------------------------
# 5. SPLIT BEFORE FITTING PREPROCESSING
# ---------------------------------------------------------------------------

if TEST_STRATEGY == "random":
    development_df, test_df = train_test_split(
        raw_df,
        test_size=TEST_FRACTION,
        random_state=RANDOM_SEED,
    )

elif TEST_STRATEGY == "high_pressure_extrapolation":
    test_df = raw_df[
        raw_df["Pressure_bar"] > EXTRAPOLATION_PRESSURE_THRESHOLD_BAR
    ].copy()

    development_df = raw_df[
        raw_df["Pressure_bar"] <= EXTRAPOLATION_PRESSURE_THRESHOLD_BAR
    ].copy()

else:
    raise ValueError(
        'TEST_STRATEGY must be "random" or "high_pressure_extrapolation".'
    )

train_df, validation_df = train_test_split(
    development_df,
    test_size=VALIDATION_FRACTION_OF_REMAINDER,
    random_state=RANDOM_SEED,
)

train_df = train_df.copy()
validation_df = validation_df.copy()
test_df = test_df.copy()


# ---------------------------------------------------------------------------
# 6. OPTIONAL OUTLIER REMOVAL
# ---------------------------------------------------------------------------

if REMOVE_FLAGGED_OUTLIERS_FROM_TRAINING:
    if "Synthetic_Outlier_Flag" in train_df.columns:
        train_df = train_df[train_df["Synthetic_Outlier_Flag"] == False].copy()


# ---------------------------------------------------------------------------
# 7. MISSING VALUES
# ---------------------------------------------------------------------------

imputation_values = {}

if MISSING_VALUE_METHOD == "drop":
    train_df = train_df.dropna(subset=RAW_FEATURE_COLUMNS)
    validation_df = validation_df.dropna(subset=RAW_FEATURE_COLUMNS)
    test_df = test_df.dropna(subset=RAW_FEATURE_COLUMNS)

elif MISSING_VALUE_METHOD == "median":

    if FIT_PREPROCESSING_ON_TRAINING_ONLY:
        imputation_source = train_df
    else:
        # This intentionally leaks information from validation/test data.
        imputation_source = pd.concat(
            [train_df, validation_df, test_df],
            ignore_index=True,
        )

    for feature in RAW_FEATURE_COLUMNS:
        replacement = float(imputation_source[feature].median())
        imputation_values[feature] = replacement

        train_df[feature] = train_df[feature].fillna(replacement)
        validation_df[feature] = validation_df[feature].fillna(replacement)
        test_df[feature] = test_df[feature].fillna(replacement)

else:
    raise ValueError('MISSING_VALUE_METHOD must be "median" or "drop".')


# ---------------------------------------------------------------------------
# 8. FEATURE ENGINEERING: LOG PRESSURE
# ---------------------------------------------------------------------------

processed_features = RAW_FEATURE_COLUMNS.copy()

if LOG_TRANSFORM_PRESSURE:
    for dataframe in [train_df, validation_df, test_df]:
        dataframe["Log10_Pressure_bar"] = np.log10(dataframe["Pressure_bar"])

    processed_features.remove("Pressure_bar")
    processed_features.append("Log10_Pressure_bar")


# ---------------------------------------------------------------------------
# 9. NORMALIZATION
# ---------------------------------------------------------------------------

normalization_mean = {}
normalization_std = {}

if NORMALIZE_FEATURES:

    if FIT_PREPROCESSING_ON_TRAINING_ONLY:
        normalization_source = train_df
    else:
        normalization_source = pd.concat(
            [train_df, validation_df, test_df],
            ignore_index=True,
        )

    for feature in processed_features:
        feature_mean = float(normalization_source[feature].mean())
        feature_std = float(normalization_source[feature].std())

        if feature_std == 0.0:
            feature_std = 1.0

        normalization_mean[feature] = feature_mean
        normalization_std[feature] = feature_std

        for dataframe in [train_df, validation_df, test_df]:
            dataframe[feature] = (
                dataframe[feature] - feature_mean
            ) / feature_std


# ---------------------------------------------------------------------------
# 10. SAVE THE PROCESSED SPLITS
# ---------------------------------------------------------------------------

# Keep raw pressure too. It is useful later for diagnosing model error.
columns_to_save = ["Sample_ID"] + processed_features + [TARGET_COLUMN, "Pressure_bar"]
columns_to_save = list(dict.fromkeys(columns_to_save))

train_file = PROCESSED_DIRECTORY / "train_processed.csv"
validation_file = PROCESSED_DIRECTORY / "validation_processed.csv"
test_file = PROCESSED_DIRECTORY / "test_processed.csv"

train_df[columns_to_save].to_csv(train_file, index=False)
validation_df[columns_to_save].to_csv(validation_file, index=False)
test_df[columns_to_save].to_csv(test_file, index=False)

metadata = {
    "RAW_FEATURE_COLUMNS": RAW_FEATURE_COLUMNS,
    "PROCESSED_FEATURE_COLUMNS": processed_features,
    "TARGET_COLUMN": TARGET_COLUMN,
    "LOG_TRANSFORM_PRESSURE": LOG_TRANSFORM_PRESSURE,
    "NORMALIZE_FEATURES": NORMALIZE_FEATURES,
    "MISSING_VALUE_METHOD": MISSING_VALUE_METHOD,
    "TEST_STRATEGY": TEST_STRATEGY,
    "FIT_PREPROCESSING_ON_TRAINING_ONLY": FIT_PREPROCESSING_ON_TRAINING_ONLY,
    "IMPUTATION_VALUES": imputation_values,
    "NORMALIZATION_MEAN": normalization_mean,
    "NORMALIZATION_STD": normalization_std,
    "TRAIN_ROWS": len(train_df),
    "VALIDATION_ROWS": len(validation_df),
    "TEST_ROWS": len(test_df),
}

metadata_file = PROCESSED_DIRECTORY / "processing_metadata.json"

with open(metadata_file, "w") as file:
    json.dump(metadata, file, indent=4)


# ---------------------------------------------------------------------------
# 11. PROCESSING DIAGNOSTIC FIGURE
# ---------------------------------------------------------------------------

figure, axes = plt.subplots(1, 2, figsize=(15, 5))

raw_training = raw_df[raw_df["Sample_ID"].isin(train_df["Sample_ID"])]
raw_box_data = [raw_training[f].dropna().to_numpy() for f in RAW_FEATURE_COLUMNS]

axes[0].boxplot(raw_box_data, tick_labels=RAW_FEATURE_COLUMNS)
axes[0].set_title("Raw Feature Scales")
axes[0].tick_params(axis="x", rotation=70)

processed_box_data = [train_df[f].to_numpy() for f in processed_features]
axes[1].boxplot(processed_box_data, tick_labels=processed_features)
axes[1].set_title("Features Given to the Neural Network")
axes[1].tick_params(axis="x", rotation=70)

figure.suptitle("Stage 2: Effect of Data Processing", fontsize=15)
figure.tight_layout()
figure.savefig(PROCESSED_DIRECTORY / "02_processing_diagnostics.png", dpi=200)
plt.show()


# ---------------------------------------------------------------------------
# 12. SUMMARY
# ---------------------------------------------------------------------------

print("Training rows:", len(train_df))
print("Validation rows:", len(validation_df))
print("Test rows:", len(test_df))
print()
print("Model features:")
print(processed_features)
print()
print("Log pressure:", LOG_TRANSFORM_PRESSURE)
print("Normalize features:", NORMALIZE_FEATURES)
print("Test strategy:", TEST_STRATEGY)
print()
print("Next: run 03_model_training/03_train_mof_neural_network.py")
