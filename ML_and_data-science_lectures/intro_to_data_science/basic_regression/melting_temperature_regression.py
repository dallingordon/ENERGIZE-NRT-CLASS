"""
10_melting_temperature_regression.py

Purpose
-------
This example uses the structured materials data created in the previous
example to train machine-learning regression models.

The goal is to predict melting temperature from several other material
properties.

The script performs the following steps:

    1. Read the 100 individual raw material CSV files.
    2. Combine them into one pandas DataFrame.
    3. Choose input features, called X.
    4. Choose melting temperature as the prediction target, called y.
    5. Train three regression models:

           - Linear regression
           - Polynomial regression
           - Kernel ridge regression (KRR)

    6. Generate 25 new synthetic materials that were not used for training.
    7. Predict the melting temperature of the 25 new materials.
    8. Compare predicted and actual melting temperatures.
    9. Save predictions, model metrics, and a six-panel parity plot.

Expected folder arrangement
---------------------------

This script assumes the previous structured-data example is stored in:

    ../structured_data/

relative to the location of this script.

For example:

    course_examples/
    |
    |-- structured_data/
    |   |
    |   |-- 09_structured_material_data.py
    |   |
    |   |-- raw_material_files/
    |       |
    |       |-- MAT_001.csv
    |       |-- MAT_002.csv
    |       |-- ...
    |
    |-- regression/
        |
        |-- 10_melting_temperature_regression.py

The input path can be changed near the top of this script if needed.

IMPORTANT
---------
The materials in this example are synthetic.

The properties were generated to be physically plausible for crystalline
metallic materials, but they do not represent specific real materials.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


# ---------------------------------------------------------------------------
# 1. INPUT AND OUTPUT PATHS
# ---------------------------------------------------------------------------

# Find the directory containing this Python script.
#
# Using the script's location makes the paths work even if the script is
# launched from a different terminal directory.

SCRIPT_DIRECTORY = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# INPUT PATH
# ---------------------------------------------------------------------------

# The previous structured-data example is assumed to be one directory above
# this script in a folder called "structured_data".
#
# The raw material CSV files are stored inside:
#
#     ../structured_data/raw_material_files/
#
# If your folders are arranged differently, this is the main path you should
# change.

RAW_DATA_DIRECTORY = (
    SCRIPT_DIRECTORY
    / ".."
    / "structured_data"
    / "raw_material_files"
).resolve()


# ---------------------------------------------------------------------------
# OUTPUT PATH
# ---------------------------------------------------------------------------

# Results from this script will be saved in a folder beside this script:
#
#     regression_results/

RESULTS_DIRECTORY = (
    SCRIPT_DIRECTORY
    / "regression_results"
)

RESULTS_DIRECTORY.mkdir(
    exist_ok=True
)


# Print the paths so students can see exactly where Python is looking.

print("Input raw-data directory:")
print(RAW_DATA_DIRECTORY)
print()

print("Output results directory:")
print(RESULTS_DIRECTORY)
print()


# ---------------------------------------------------------------------------
# 2. FIND THE RAW MATERIAL FILES
# ---------------------------------------------------------------------------

# Each raw file contains one material.
#
# The files have names such as:
#
# MAT_001.csv
# MAT_002.csv
# MAT_003.csv

raw_files = sorted(
    RAW_DATA_DIRECTORY.glob("MAT_*.csv")
)

print(
    "Number of training material files found:",
    len(raw_files)
)
print()


# Stop with a useful error if the raw files are missing.

if len(raw_files) == 0:
    raise FileNotFoundError(
        "\nNo raw material files were found.\n\n"
        "Python looked in:\n"
        f"{RAW_DATA_DIRECTORY}\n\n"
        "Check that the previous structured-data example is stored in:\n"
        "../structured_data/\n"
        "relative to this script."
    )


# ---------------------------------------------------------------------------
# 3. READ ALL OF THE RAW FILES
# ---------------------------------------------------------------------------

# We first read every one-row material file into a small DataFrame.
#
# We store those DataFrames in a Python list.

material_dataframes = []

for raw_file in raw_files:

    # Read one CSV file.
    material_df = pd.read_csv(
        raw_file
    )

    # Store the small DataFrame in our list.
    material_dataframes.append(
        material_df
    )


# Combine all of the one-row DataFrames into one large DataFrame.

training_df = pd.concat(
    material_dataframes,
    ignore_index=True
)


print("Combined training DataFrame:")
print(training_df.head())
print()

print("Training DataFrame shape:")
print(training_df.shape)
print()


# ---------------------------------------------------------------------------
# 4. DEFINE THE INPUT FEATURES
# ---------------------------------------------------------------------------

# A feature is a piece of information used by a model to make a prediction.
#
# Each material is represented by five numerical features.
#
# Melting temperature is NOT included because it is the quantity we want
# the models to predict.

FEATURE_COLUMNS = [
    "Vacancy_Formation_Energy_eV",
    "Bulk_Modulus_GPa",
    "Average_Surface_Energy_J_m2",
    "Stacking_Fault_Energy_mJ_m2",
    "Average_Bond_Strength_eV",
]

TARGET_COLUMN = "Melting_Temperature_K"


# ---------------------------------------------------------------------------
# 5. CREATE X AND y
# ---------------------------------------------------------------------------

# In machine learning:
#
# X = input features
# y = target value we want to predict

X_train = training_df[
    FEATURE_COLUMNS
]

y_train = training_df[
    TARGET_COLUMN
]


print("Feature names:")
print(FEATURE_COLUMNS)
print()

print("Shape of X_train:")
print(X_train.shape)
print()

print("First material feature vector:")
print(X_train.iloc[0])
print()

print("First material melting temperature:")
print(
    y_train.iloc[0],
    "K"
)
print()


# ---------------------------------------------------------------------------
# 6. MODEL 1: LINEAR REGRESSION
# ---------------------------------------------------------------------------

# Linear regression assumes that the target can be approximated as a
# weighted sum of the input features.

linear_model = LinearRegression()


# ---------------------------------------------------------------------------
# 7. MODEL 2: POLYNOMIAL REGRESSION
# ---------------------------------------------------------------------------

# Polynomial regression can represent curved relationships.
#
# degree=2 creates terms such as:
#
# x1
# x2
# x1^2
# x2^2
# x1*x2
#
# StandardScaler places the features on more similar numerical scales.

polynomial_model = Pipeline(
    steps=[
        (
            "scaler",
            StandardScaler()
        ),
        (
            "polynomial_features",
            PolynomialFeatures(
                degree=2,
                include_bias=False
            )
        ),
        (
            "linear_regression",
            LinearRegression()
        ),
    ]
)


# ---------------------------------------------------------------------------
# 8. MODEL 3: KERNEL RIDGE REGRESSION
# ---------------------------------------------------------------------------

# Kernel ridge regression can learn nonlinear relationships.
#
# The RBF kernel compares how similar different feature vectors are.
#
# alpha controls regularization.
# gamma controls the width of the RBF kernel.
#
# These values are fixed here to keep this introductory example simple.

krr_model = Pipeline(
    steps=[
        (
            "scaler",
            StandardScaler()
        ),
        (
            "kernel_ridge",
            KernelRidge(
                kernel="rbf",
                alpha=0.01,
                gamma=0.01
            )
        ),
    ]
)


# ---------------------------------------------------------------------------
# 9. STORE THE MODELS TOGETHER
# ---------------------------------------------------------------------------

models = {
    "Linear Regression": linear_model,
    "Polynomial Regression": polynomial_model,
    "Kernel Ridge Regression": krr_model,
}


# ---------------------------------------------------------------------------
# 10. TRAIN ALL THREE MODELS
# ---------------------------------------------------------------------------

# fit() is the scikit-learn command used to train a model.

for model_name, model in models.items():

    model.fit(
        X_train,
        y_train
    )

    print(
        model_name,
        "training complete."
    )

print()


# ---------------------------------------------------------------------------
# 11. MAKE PREDICTIONS FOR THE TRAINING DATA
# ---------------------------------------------------------------------------

training_predictions = {}

for model_name, model in models.items():

    predicted_temperature = model.predict(
        X_train
    )

    training_predictions[
        model_name
    ] = predicted_temperature


# ---------------------------------------------------------------------------
# 12. GENERATE 25 NEW MATERIALS
# ---------------------------------------------------------------------------

# These new materials are generated independently from the 100 training
# materials.
#
# A different random seed ensures that the feature vectors are different.

NUMBER_OF_NEW_MATERIALS = 25

NEW_DATA_RANDOM_SEED = 2026

rng = np.random.default_rng(
    NEW_DATA_RANDOM_SEED
)

new_material_records = []


# Spread the new materials across a broad range of bond strengths so the
# test set includes weakly bonded and strongly bonded examples.

bond_strength_values = np.linspace(
    1.4,
    4.3,
    NUMBER_OF_NEW_MATERIALS
)


# Add a small random shift so the values are not perfectly evenly spaced.

bond_strength_values = (
    bond_strength_values
    + rng.normal(
        loc=0.0,
        scale=0.05,
        size=NUMBER_OF_NEW_MATERIALS
    )
)


for material_number, average_bond_strength in enumerate(
    bond_strength_values,
    start=1
):

    new_material_id = (
        f"NEW_{material_number:03d}"
    )

    average_bond_strength = np.clip(
        average_bond_strength,
        1.3,
        4.5
    )


    # -----------------------------------------------------------------------
    # Vacancy formation energy
    # -----------------------------------------------------------------------

    vacancy_formation_energy = (
        0.35
        + 0.70 * average_bond_strength
        + rng.normal(
            loc=0.0,
            scale=0.18
        )
    )

    vacancy_formation_energy = np.clip(
        vacancy_formation_energy,
        0.8,
        3.8
    )


    # -----------------------------------------------------------------------
    # Bulk modulus
    # -----------------------------------------------------------------------

    bulk_modulus = (
        20.0
        + 52.0 * average_bond_strength
        + rng.normal(
            loc=0.0,
            scale=15.0
        )
    )

    bulk_modulus = np.clip(
        bulk_modulus,
        45.0,
        285.0
    )


    # -----------------------------------------------------------------------
    # Average surface energy
    # -----------------------------------------------------------------------

    average_surface_energy = (
        0.30
        + 0.68 * average_bond_strength
        + rng.normal(
            loc=0.0,
            scale=0.20
        )
    )

    average_surface_energy = np.clip(
        average_surface_energy,
        0.8,
        3.6
    )


    # -----------------------------------------------------------------------
    # Stacking fault energy
    # -----------------------------------------------------------------------

    defect_character = rng.uniform(
        0.0,
        1.0
    )

    stacking_fault_energy = (
        20.0
        + 28.0 * average_bond_strength
        + 185.0 * defect_character
        + rng.normal(
            loc=0.0,
            scale=20.0
        )
    )

    stacking_fault_energy = np.clip(
        stacking_fault_energy,
        25.0,
        350.0
    )


    # -----------------------------------------------------------------------
    # True melting temperature
    # -----------------------------------------------------------------------

    # We retain the true synthetic melting temperature so that we can compare
    # the predictions with a known answer.
    #
    # In a real application, the true value could come from experiment or
    # simulation.

    melting_temperature = (
        380.0
        + 470.0 * average_bond_strength
        + rng.normal(
            loc=0.0,
            scale=130.0
        )
    )

    melting_temperature = np.clip(
        melting_temperature,
        700.0,
        2800.0
    )


    # Store all properties for this new material.

    new_material_records.append(
        {
            "Material_ID":
                new_material_id,

            "Vacancy_Formation_Energy_eV":
                vacancy_formation_energy,

            "Bulk_Modulus_GPa":
                bulk_modulus,

            "Average_Surface_Energy_J_m2":
                average_surface_energy,

            "Stacking_Fault_Energy_mJ_m2":
                stacking_fault_energy,

            "Average_Bond_Strength_eV":
                average_bond_strength,

            "Melting_Temperature_K":
                melting_temperature,
        }
    )


# Convert the new materials into a pandas DataFrame.

new_materials_df = pd.DataFrame(
    new_material_records
)

new_materials_df = new_materials_df.round(
    3
)


print("First five new materials:")
print(new_materials_df.head())
print()


# ---------------------------------------------------------------------------
# 13. CREATE FEATURE VECTORS FOR THE 25 NEW MATERIALS
# ---------------------------------------------------------------------------

X_new = new_materials_df[
    FEATURE_COLUMNS
]

y_new = new_materials_df[
    TARGET_COLUMN
]


# ---------------------------------------------------------------------------
# 14. PREDICT THE NEW MELTING TEMPERATURES
# ---------------------------------------------------------------------------

new_predictions = {}

for model_name, model in models.items():

    predicted_temperature = model.predict(
        X_new
    )

    new_predictions[
        model_name
    ] = predicted_temperature


# ---------------------------------------------------------------------------
# 15. CREATE A TABLE OF THE NEW PREDICTIONS
# ---------------------------------------------------------------------------

prediction_df = new_materials_df.copy()


prediction_df[
    "Linear_Regression_Predicted_T_K"
] = new_predictions[
    "Linear Regression"
]


prediction_df[
    "Polynomial_Regression_Predicted_T_K"
] = new_predictions[
    "Polynomial Regression"
]


prediction_df[
    "KRR_Predicted_T_K"
] = new_predictions[
    "Kernel Ridge Regression"
]


# Round the table so it is easier to read.

prediction_df = prediction_df.round(
    2
)


# ---------------------------------------------------------------------------
# 16. SAVE THE NEW-MATERIAL PREDICTIONS
# ---------------------------------------------------------------------------

prediction_csv = (
    RESULTS_DIRECTORY
    / "new_material_melting_temperature_predictions.csv"
)

prediction_excel = (
    RESULTS_DIRECTORY
    / "new_material_melting_temperature_predictions.xlsx"
)


prediction_df.to_csv(
    prediction_csv,
    index=False
)

prediction_df.to_excel(
    prediction_excel,
    index=False
)


print("New-material predictions:")
print(
    prediction_df[
        [
            "Material_ID",
            "Melting_Temperature_K",
            "Linear_Regression_Predicted_T_K",
            "Polynomial_Regression_Predicted_T_K",
            "KRR_Predicted_T_K",
        ]
    ]
)
print()


# ---------------------------------------------------------------------------
# 17. CALCULATE MODEL PERFORMANCE
# ---------------------------------------------------------------------------

# We calculate:
#
# R^2
#     Values closer to 1 indicate better agreement.
#
# MAE
#     Mean Absolute Error is the average absolute prediction error.
#     Its units are Kelvin.

metric_records = []


for model_name in models:

    train_prediction = training_predictions[
        model_name
    ]

    new_prediction = new_predictions[
        model_name
    ]


    train_r2 = r2_score(
        y_train,
        train_prediction
    )

    train_mae = mean_absolute_error(
        y_train,
        train_prediction
    )

    new_r2 = r2_score(
        y_new,
        new_prediction
    )

    new_mae = mean_absolute_error(
        y_new,
        new_prediction
    )


    metric_records.append(
        {
            "Model":
                model_name,

            "Training_R2":
                train_r2,

            "Training_MAE_K":
                train_mae,

            "New_Data_R2":
                new_r2,

            "New_Data_MAE_K":
                new_mae,
        }
    )


metrics_df = pd.DataFrame(
    metric_records
)

metrics_df = metrics_df.round(
    3
)


metrics_file = (
    RESULTS_DIRECTORY
    / "regression_model_metrics.csv"
)

metrics_df.to_csv(
    metrics_file,
    index=False
)


print("Model performance:")
print(metrics_df)
print()


# ---------------------------------------------------------------------------
# 18. CREATE THE SIX-PANEL PARITY PLOT
# ---------------------------------------------------------------------------

# A parity plot compares the true value with the predicted value.
#
# Perfect predictions would lie directly on the diagonal y = x line.
#
# Layout:
#
# TOP ROW
#
#     Linear Regression
#     Polynomial Regression
#     Kernel Ridge Regression
#
# BOTTOM ROW
#
#     The same three models evaluated on the 25 new materials.

figure, axes = plt.subplots(
    nrows=2,
    ncols=3,
    figsize=(15, 9)
)


# Find one common axis range for all six plots.
#
# Using the same limits makes visual comparison easier.

all_values = [
    y_train.to_numpy(),
    y_new.to_numpy(),
]


for model_name in models:

    all_values.append(
        training_predictions[
            model_name
        ]
    )

    all_values.append(
        new_predictions[
            model_name
        ]
    )


all_values = np.concatenate(
    all_values
)


plot_minimum = (
    np.min(all_values)
    - 100.0
)

plot_maximum = (
    np.max(all_values)
    + 100.0
)


# Create one column for each model.

for column_number, model_name in enumerate(
    models
):

    # -----------------------------------------------------------------------
    # TOP ROW: TRAINING DATA
    # -----------------------------------------------------------------------

    train_axis = axes[
        0,
        column_number
    ]


    train_axis.scatter(
        y_train,
        training_predictions[
            model_name
        ]
    )


    # Draw the perfect-prediction line.

    train_axis.plot(
        [
            plot_minimum,
            plot_maximum
        ],
        [
            plot_minimum,
            plot_maximum
        ],
        linestyle="--"
    )


    train_axis.set_xlim(
        plot_minimum,
        plot_maximum
    )

    train_axis.set_ylim(
        plot_minimum,
        plot_maximum
    )


    train_axis.set_xlabel(
        "True Melting Temperature (K)"
    )

    train_axis.set_ylabel(
        "Predicted Melting Temperature (K)"
    )


    train_r2 = r2_score(
        y_train,
        training_predictions[
            model_name
        ]
    )

    train_mae = mean_absolute_error(
        y_train,
        training_predictions[
            model_name
        ]
    )


    train_axis.set_title(
        model_name
        + "\nTraining Data"
        + f"\nR² = {train_r2:.3f}, "
        + f"MAE = {train_mae:.1f} K"
    )


    # -----------------------------------------------------------------------
    # BOTTOM ROW: 25 NEW MATERIALS
    # -----------------------------------------------------------------------

    new_axis = axes[
        1,
        column_number
    ]


    new_axis.scatter(
        y_new,
        new_predictions[
            model_name
        ]
    )


    new_axis.plot(
        [
            plot_minimum,
            plot_maximum
        ],
        [
            plot_minimum,
            plot_maximum
        ],
        linestyle="--"
    )


    new_axis.set_xlim(
        plot_minimum,
        plot_maximum
    )

    new_axis.set_ylim(
        plot_minimum,
        plot_maximum
    )


    new_axis.set_xlabel(
        "True Melting Temperature (K)"
    )

    new_axis.set_ylabel(
        "Predicted Melting Temperature (K)"
    )


    new_r2 = r2_score(
        y_new,
        new_predictions[
            model_name
        ]
    )

    new_mae = mean_absolute_error(
        y_new,
        new_predictions[
            model_name
        ]
    )


    new_axis.set_title(
        model_name
        + "\n25 New Materials"
        + f"\nR² = {new_r2:.3f}, "
        + f"MAE = {new_mae:.1f} K"
    )


figure.suptitle(
    "Melting Temperature Regression",
    fontsize=16
)


figure.tight_layout()


plot_file = (
    RESULTS_DIRECTORY
    / "melting_temperature_parity_plots.png"
)


figure.savefig(
    plot_file,
    dpi=200
)


plt.show()


# ---------------------------------------------------------------------------
# 19. FINAL OUTPUT SUMMARY
# ---------------------------------------------------------------------------

print("Analysis complete.")
print()

print("Input data were read from:")
print(RAW_DATA_DIRECTORY)
print()

print("Results were saved in:")
print(RESULTS_DIRECTORY)
print()

print("Prediction table:")
print(prediction_csv)
print()

print("Model metrics:")
print(metrics_file)
print()

print("Parity plot:")
print(plot_file)
print()


# ---------------------------------------------------------------------------
# PRACTICE EXERCISES
# ---------------------------------------------------------------------------

# Try the following:
#
# 1. Remove one feature from FEATURE_COLUMNS.
#    Train the models again.
#    Does prediction quality improve or become worse?
#
# 2. Change the polynomial degree from 2 to 3.
#    What happens to the training performance?
#    What happens to the 25 new materials?
#
# 3. Change the KRR gamma value.
#
#    Try:
#
#        gamma = 0.001
#        gamma = 0.01
#        gamma = 0.1
#        gamma = 1.0
#
#    How does this affect the parity plots?
#
# 4. Look at correlations using:
#
#        training_df.corr(numeric_only=True)
#
#    Which feature is most strongly related to melting temperature?
#
# 5. Sort prediction_df by the absolute prediction error for one model.
#    Which new materials are the hardest to predict?
#
# 6. Change RAW_DATA_DIRECTORY near the top of the script so that the
#    program reads data from another folder.
