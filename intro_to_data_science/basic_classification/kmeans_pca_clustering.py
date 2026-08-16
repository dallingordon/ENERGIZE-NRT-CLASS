"""
11_kmeans_pca_clustering.py

Purpose
-------
This example introduces three important ideas used in materials data science:

    1. Feature scaling
    2. Principal component analysis (PCA)
    3. KMeans clustering

The script first uses the 100 synthetic training materials created in the
structured-data example.

Each material is represented by five input features:

    - Vacancy formation energy
    - Bulk modulus
    - Average surface energy
    - Stacking fault energy
    - Average bond strength

Melting temperature is NOT used to create the PCA space or the KMeans
clusters.

The workflow has two parts.

PART 1: UNSUPERVISED CLUSTERING

    1. Read the training materials.
    2. Standardize the five input features.
    3. Reduce the five-dimensional feature vectors to two dimensions using PCA.
    4. Fit a KMeans model in the two-dimensional PCA space.
    5. Plot the PCA space colored by:
           - KMeans cluster
           - Melting temperature

PART 2: CLASSIFYING NEW MATERIALS

    1. Read the 25 new materials created in the basic regression example.
    2. Transform them using the SAME scaler and PCA model used for training.
    3. Find the k nearest training materials in the PCA space.
    4. Assign each new material to a cluster using a majority vote of the
       neighboring training materials.
    5. Estimate melting temperature using the average melting temperature of
       those same nearby training materials.
    6. Plot where the new materials fall in the PCA space.

Expected folder arrangement
---------------------------

This script assumes a folder structure similar to:

    course_examples/
    |
    |-- structured_data/
    |   |
    |   |-- raw_material_files/
    |       |
    |       |-- MAT_001.csv
    |       |-- MAT_002.csv
    |       |-- ...
    |
    |-- basic_regression/
    |   |
    |   |-- regression_results/
    |       |
    |       |-- new_material_melting_temperature_predictions.csv
    |
    |-- clustering/
        |
        |-- 11_kmeans_pca_clustering.py

The input paths are defined near the top of this script and can be changed
if your folders are arranged differently.

IMPORTANT
---------
The materials in this example are synthetic and do not represent specific
real materials.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


# ---------------------------------------------------------------------------
# 1. INPUT AND OUTPUT PATHS
# ---------------------------------------------------------------------------

# Find the directory containing this Python script.
#
# Building paths from the script location makes the example more reliable
# than assuming the terminal was opened in a particular folder.

SCRIPT_DIRECTORY = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# TRAINING DATA PATH
# ---------------------------------------------------------------------------

# The 100 raw training material files are assumed to be in:
#
#     ../structured_data/raw_material_files/

TRAINING_RAW_DATA_DIRECTORY = (
    SCRIPT_DIRECTORY
    / ".."
    / "structured_data"
    / "raw_material_files"
).resolve()


# ---------------------------------------------------------------------------
# TEST DATA PATH
# ---------------------------------------------------------------------------

# The previous regression example saved the 25 new materials in:
#
#     ../basic_regression/regression_results/
#
# The file contains the original feature vectors, the true synthetic melting
# temperatures, and the regression-model predictions.
#
# This clustering example only uses the original feature columns and the
# true melting temperature column.

TEST_DATA_FILE = (
    SCRIPT_DIRECTORY
    / ".."
    / "basic_regression"
    / "regression_results"
    / "new_material_melting_temperature_predictions.csv"
).resolve()


# ---------------------------------------------------------------------------
# OUTPUT PATH
# ---------------------------------------------------------------------------

RESULTS_DIRECTORY = (
    SCRIPT_DIRECTORY
    / "clustering_results"
)

RESULTS_DIRECTORY.mkdir(
    exist_ok=True
)


print("Training data directory:")
print(TRAINING_RAW_DATA_DIRECTORY)
print()

print("Test data file:")
print(TEST_DATA_FILE)
print()

print("Results directory:")
print(RESULTS_DIRECTORY)
print()


# ---------------------------------------------------------------------------
# 2. SETTINGS
# ---------------------------------------------------------------------------

# Number of KMeans clusters.
#
# Four clusters keeps the example visually simple while still showing that
# the materials can occupy several regions of feature space.

NUMBER_OF_CLUSTERS = 4


# Number of nearby training materials used when classifying each new material.
#
# A small odd number is useful because it reduces the chance of a tied vote.

K_NEIGHBORS = 5


# Random seed used by KMeans.
#
# Setting a seed makes the clustering reproducible.

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# 3. FEATURE AND TARGET NAMES
# ---------------------------------------------------------------------------

# These are the five features used to represent each material.
#
# Melting temperature is deliberately left out.

FEATURE_COLUMNS = [
    "Vacancy_Formation_Energy_eV",
    "Bulk_Modulus_GPa",
    "Average_Surface_Energy_J_m2",
    "Stacking_Fault_Energy_mJ_m2",
    "Average_Bond_Strength_eV",
]

TARGET_COLUMN = "Melting_Temperature_K"


# ---------------------------------------------------------------------------
# 4. FIND AND READ THE TRAINING FILES
# ---------------------------------------------------------------------------

training_files = sorted(
    TRAINING_RAW_DATA_DIRECTORY.glob("MAT_*.csv")
)

print(
    "Number of training material files found:",
    len(training_files)
)
print()


if len(training_files) == 0:

    raise FileNotFoundError(
        "\nNo training material files were found.\n\n"
        "Python looked in:\n"
        f"{TRAINING_RAW_DATA_DIRECTORY}\n\n"
        "Check that the structured-data example is stored in:\n"
        "../structured_data/\n"
        "relative to this script."
    )


training_dataframes = []


for training_file in training_files:

    material_df = pd.read_csv(
        training_file
    )

    training_dataframes.append(
        material_df
    )


training_df = pd.concat(
    training_dataframes,
    ignore_index=True
)


print("First five training materials:")
print(training_df.head())
print()

print("Training DataFrame shape:")
print(training_df.shape)
print()


# ---------------------------------------------------------------------------
# 5. CREATE THE TRAINING FEATURE MATRIX
# ---------------------------------------------------------------------------

# X_train contains only the five input features.
#
# y_train contains melting temperature.
#
# Melting temperature is kept separate and is NOT used by PCA or KMeans.

X_train = training_df[
    FEATURE_COLUMNS
]

y_train = training_df[
    TARGET_COLUMN
]


print("Training feature matrix shape:")
print(X_train.shape)
print()


# ---------------------------------------------------------------------------
# 6. STANDARDIZE THE FEATURES
# ---------------------------------------------------------------------------

# The five material properties have very different numerical scales.
#
# For example:
#
# Vacancy formation energy might be around 1 to 4 eV.
# Bulk modulus might be around 50 to 300 GPa.
# Stacking fault energy might be around 20 to 350 mJ/m^2.
#
# PCA is affected by numerical scale, so we standardize every feature first.
#
# StandardScaler transforms each feature so that it has approximately:
#
#     mean = 0
#     standard deviation = 1

scaler = StandardScaler()


# fit_transform() does two things:
#
# 1. Learns the mean and standard deviation from the TRAINING data.
# 2. Uses them to scale the training data.

X_train_scaled = scaler.fit_transform(
    X_train
)


# ---------------------------------------------------------------------------
# 7. REDUCE THE FEATURE VECTORS TO TWO DIMENSIONS USING PCA
# ---------------------------------------------------------------------------

# The original feature vectors have five dimensions.
#
# PCA finds new directions through the data called principal components.
#
# We keep only two components so that the materials can be displayed on a
# normal two-dimensional plot.

pca = PCA(
    n_components=2
)


X_train_pca = pca.fit_transform(
    X_train_scaled
)


# Print how much information is captured by each PCA dimension.

explained_variance = pca.explained_variance_ratio_


print("PCA explained variance:")
print(
    f"PC1: {explained_variance[0] * 100:.1f}%"
)
print(
    f"PC2: {explained_variance[1] * 100:.1f}%"
)
print(
    "Total explained variance:",
    f"{explained_variance.sum() * 100:.1f}%"
)
print()


# ---------------------------------------------------------------------------
# 8. FIT KMEANS IN THE TWO-DIMENSIONAL PCA SPACE
# ---------------------------------------------------------------------------

# KMeans groups nearby points into a chosen number of clusters.
#
# KMeans does NOT use melting temperature here.
#
# It only sees the two PCA coordinates.

kmeans = KMeans(
    n_clusters=NUMBER_OF_CLUSTERS,
    random_state=RANDOM_SEED,
    n_init=10
)


training_cluster_labels = kmeans.fit_predict(
    X_train_pca
)


print("Number of training materials in each cluster:")

for cluster_number in range(
    NUMBER_OF_CLUSTERS
):

    number_in_cluster = np.sum(
        training_cluster_labels
        == cluster_number
    )

    print(
        f"Cluster {cluster_number}:",
        number_in_cluster
    )

print()


# ---------------------------------------------------------------------------
# 9. SAVE THE TRAINING PCA AND CLUSTER INFORMATION
# ---------------------------------------------------------------------------

training_results_df = training_df.copy()


training_results_df[
    "PCA_1"
] = X_train_pca[:, 0]


training_results_df[
    "PCA_2"
] = X_train_pca[:, 1]


training_results_df[
    "KMeans_Cluster"
] = training_cluster_labels


training_results_df = training_results_df.round(
    4
)


training_results_file = (
    RESULTS_DIRECTORY
    / "training_materials_pca_clusters.csv"
)


training_results_df.to_csv(
    training_results_file,
    index=False
)


# ---------------------------------------------------------------------------
# 10. UNSUPERVISED CLUSTERING FIGURE
# ---------------------------------------------------------------------------

# The left subplot shows the PCA space colored by KMeans cluster.
#
# The right subplot shows the exact same points, but colored by melting
# temperature.
#
# Comparing these plots helps us see whether the unsupervised clusters also
# separate materials according to melting temperature.

figure, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(12, 5)
)


# ---------------------------------------------------------------------------
# LEFT: KMEANS CLUSTERS
# ---------------------------------------------------------------------------

cluster_scatter = axes[0].scatter(
    X_train_pca[:, 0],
    X_train_pca[:, 1],
    c=training_cluster_labels,
    cmap="tab10",
    s=60
)


axes[0].set_xlabel(
    "Principal Component 1"
)

axes[0].set_ylabel(
    "Principal Component 2"
)

axes[0].set_title(
    "Training Materials Colored by KMeans Cluster"
)


cluster_colorbar = figure.colorbar(
    cluster_scatter,
    ax=axes[0]
)

cluster_colorbar.set_label(
    "KMeans Cluster"
)


# ---------------------------------------------------------------------------
# RIGHT: MELTING TEMPERATURE
# ---------------------------------------------------------------------------

temperature_scatter = axes[1].scatter(
    X_train_pca[:, 0],
    X_train_pca[:, 1],
    c=y_train,
    cmap="viridis",
    s=60
)


axes[1].set_xlabel(
    "Principal Component 1"
)

axes[1].set_ylabel(
    "Principal Component 2"
)

axes[1].set_title(
    "Training Materials Colored by Melting Temperature"
)


temperature_colorbar = figure.colorbar(
    temperature_scatter,
    ax=axes[1]
)

temperature_colorbar.set_label(
    "Melting Temperature (K)"
)


figure.suptitle(
    "Unsupervised PCA and KMeans Clustering",
    fontsize=14
)


figure.tight_layout()


unsupervised_plot_file = (
    RESULTS_DIRECTORY
    / "unsupervised_pca_kmeans.png"
)


figure.savefig(
    unsupervised_plot_file,
    dpi=200
)


plt.show()


# ---------------------------------------------------------------------------
# 11. READ THE 25 NEW TEST MATERIALS
# ---------------------------------------------------------------------------

if not TEST_DATA_FILE.exists():

    raise FileNotFoundError(
        "\nThe test-material file was not found.\n\n"
        "Python looked for:\n"
        f"{TEST_DATA_FILE}\n\n"
        "Run the basic regression example first or update TEST_DATA_FILE "
        "near the top of this script."
    )


test_df = pd.read_csv(
    TEST_DATA_FILE
)


print("Number of new test materials:")
print(len(test_df))
print()

print("First five test materials:")
print(test_df.head())
print()


# ---------------------------------------------------------------------------
# 12. CREATE TEST FEATURE VECTORS
# ---------------------------------------------------------------------------

X_test = test_df[
    FEATURE_COLUMNS
]


# The previous synthetic example also gives us the true melting temperatures.
#
# We will use these ONLY after making predictions so that we can evaluate how
# well the neighbor-based estimate worked.

y_test = test_df[
    TARGET_COLUMN
]


# ---------------------------------------------------------------------------
# 13. PROJECT THE TEST MATERIALS INTO THE EXISTING PCA SPACE
# ---------------------------------------------------------------------------

# This is extremely important:
#
# We DO NOT fit a new scaler.
# We DO NOT fit a new PCA model.
#
# We use the scaler and PCA model learned from the training data.
#
# This places the new materials into the same coordinate system as the
# original 100 materials.

X_test_scaled = scaler.transform(
    X_test
)


X_test_pca = pca.transform(
    X_test_scaled
)


# ---------------------------------------------------------------------------
# 14. FIND THE K NEAREST TRAINING MATERIALS
# ---------------------------------------------------------------------------

# NearestNeighbors will search for nearby points in the two-dimensional
# PCA space.

neighbor_model = NearestNeighbors(
    n_neighbors=K_NEIGHBORS
)


neighbor_model.fit(
    X_train_pca
)


# kneighbors() returns:
#
# distances
#     Distance from each new material to its nearby training materials.
#
# neighbor_indices
#     Row numbers of those nearby training materials.

distances, neighbor_indices = neighbor_model.kneighbors(
    X_test_pca
)


# ---------------------------------------------------------------------------
# 15. ASSIGN TEST CLUSTERS AND ESTIMATE MELTING TEMPERATURE
# ---------------------------------------------------------------------------

assigned_test_clusters = []

estimated_test_temperatures = []


for test_material_number in range(
    len(test_df)
):

    # Get the row numbers of this material's nearest training neighbors.

    nearby_indices = neighbor_indices[
        test_material_number
    ]


    # Find the KMeans cluster of every nearby training material.

    nearby_clusters = training_cluster_labels[
        nearby_indices
    ]


    # -----------------------------------------------------------------------
    # MAJORITY VOTE FOR CLUSTER ASSIGNMENT
    # -----------------------------------------------------------------------

    # Count how many nearby materials belong to each cluster.

    cluster_numbers, cluster_counts = np.unique(
        nearby_clusters,
        return_counts=True
    )


    largest_count = np.max(
        cluster_counts
    )


    # There can occasionally be a tie.
    #
    # If there is a tie, choose the tied cluster belonging to the closest
    # neighbor among those tied clusters.

    tied_clusters = cluster_numbers[
        cluster_counts == largest_count
    ]


    if len(tied_clusters) == 1:

        assigned_cluster = tied_clusters[0]

    else:

        assigned_cluster = None

        for nearby_cluster in nearby_clusters:

            if nearby_cluster in tied_clusters:

                assigned_cluster = nearby_cluster

                break


    assigned_test_clusters.append(
        assigned_cluster
    )


    # -----------------------------------------------------------------------
    # ESTIMATE MELTING TEMPERATURE
    # -----------------------------------------------------------------------

    # Look up the known melting temperatures of the same nearby training
    # materials.

    nearby_temperatures = y_train.iloc[
        nearby_indices
    ]


    # Use their average melting temperature as the prediction.

    estimated_temperature = nearby_temperatures.mean()


    estimated_test_temperatures.append(
        estimated_temperature
    )


# Convert the Python lists into NumPy arrays.

assigned_test_clusters = np.array(
    assigned_test_clusters
)


estimated_test_temperatures = np.array(
    estimated_test_temperatures
)


# ---------------------------------------------------------------------------
# 16. SAVE THE TEST-MATERIAL RESULTS
# ---------------------------------------------------------------------------

test_results_df = test_df[
    [
        "Material_ID",
        *FEATURE_COLUMNS,
        TARGET_COLUMN,
    ]
].copy()


test_results_df[
    "PCA_1"
] = X_test_pca[:, 0]


test_results_df[
    "PCA_2"
] = X_test_pca[:, 1]


test_results_df[
    "Assigned_Cluster"
] = assigned_test_clusters


test_results_df[
    "KNN_Estimated_Melting_Temperature_K"
] = estimated_test_temperatures


test_results_df[
    "Absolute_Error_K"
] = np.abs(
    estimated_test_temperatures
    - y_test.to_numpy()
)


test_results_df = test_results_df.round(
    3
)


test_results_csv = (
    RESULTS_DIRECTORY
    / "test_material_cluster_and_temperature_predictions.csv"
)


test_results_excel = (
    RESULTS_DIRECTORY
    / "test_material_cluster_and_temperature_predictions.xlsx"
)


test_results_df.to_csv(
    test_results_csv,
    index=False
)


test_results_df.to_excel(
    test_results_excel,
    index=False
)


# ---------------------------------------------------------------------------
# 17. EVALUATE THE MELTING-TEMPERATURE ESTIMATES
# ---------------------------------------------------------------------------

test_mae = mean_absolute_error(
    y_test,
    estimated_test_temperatures
)


test_r2 = r2_score(
    y_test,
    estimated_test_temperatures
)


print("Nearest-neighbor melting-temperature estimate:")
print(
    f"MAE = {test_mae:.1f} K"
)
print(
    f"R² = {test_r2:.3f}"
)
print()


# ---------------------------------------------------------------------------
# 18. NEW-MATERIAL CLASSIFICATION FIGURE
# ---------------------------------------------------------------------------

# This final figure contains THREE columns.
#
# LEFT SUBPLOT
#
# Show the original training PCA space colored by KMeans cluster.
# Overlay the 25 new materials as yellow stars with black outlines.
#
# MIDDLE SUBPLOT
#
# Show only the 25 new materials.
# Color each star according to the cluster assigned by the nearest-neighbor
# majority vote.
#
# RIGHT SUBPLOT
#
# Show only the same 25 new materials again.
# Color each star according to its TRUE melting temperature.
#
# The middle and right panels can be compared to see whether the cluster
# assignments correspond to regions with similar melting temperature.

figure, axes = plt.subplots(
    nrows=1,
    ncols=3,
    figsize=(18, 5)
)


# ---------------------------------------------------------------------------
# LEFT: TRAINING SPACE + NEW MATERIAL LOCATIONS
# ---------------------------------------------------------------------------

training_scatter = axes[0].scatter(
    X_train_pca[:, 0],
    X_train_pca[:, 1],
    c=training_cluster_labels,
    cmap="tab10",
    s=55,
    alpha=0.75
)

axes[0].scatter(
    X_test_pca[:, 0],
    X_test_pca[:, 1],
    c="yellow",
    marker="*",
    s=180,
    edgecolors="black",
    linewidths=1.0,
    label="New materials"
)

axes[0].set_xlabel(
    "Principal Component 1"
)

axes[0].set_ylabel(
    "Principal Component 2"
)

axes[0].set_title(
    "Training PCA Space with New Materials"
)

axes[0].legend()

training_colorbar = figure.colorbar(
    training_scatter,
    ax=axes[0]
)

training_colorbar.set_label(
    "Training KMeans Cluster"
)


# ---------------------------------------------------------------------------
# MIDDLE: NEW MATERIALS COLORED BY ASSIGNED CLUSTER
# ---------------------------------------------------------------------------

test_cluster_scatter = axes[1].scatter(
    X_test_pca[:, 0],
    X_test_pca[:, 1],
    c=assigned_test_clusters,
    cmap="tab10",
    marker="*",
    s=180,
    edgecolors="black",
    linewidths=1.0
)

axes[1].set_xlabel(
    "Principal Component 1"
)

axes[1].set_ylabel(
    "Principal Component 2"
)

axes[1].set_title(
    "New Materials Colored by Assigned Cluster"
)

test_cluster_colorbar = figure.colorbar(
    test_cluster_scatter,
    ax=axes[1]
)

test_cluster_colorbar.set_label(
    "Assigned Cluster"
)


# ---------------------------------------------------------------------------
# RIGHT: NEW MATERIALS COLORED BY TRUE MELTING TEMPERATURE
# ---------------------------------------------------------------------------

test_temperature_scatter = axes[2].scatter(
    X_test_pca[:, 0],
    X_test_pca[:, 1],
    c=y_test,
    cmap="viridis",
    marker="*",
    s=180,
    edgecolors="black",
    linewidths=1.0
)

axes[2].set_xlabel(
    "Principal Component 1"
)

axes[2].set_ylabel(
    "Principal Component 2"
)

axes[2].set_title(
    "New Materials Colored by True Melting Temperature"
)

test_temperature_colorbar = figure.colorbar(
    test_temperature_scatter,
    ax=axes[2]
)

test_temperature_colorbar.set_label(
    "True Melting Temperature (K)"
)


figure.suptitle(
    "Nearest-Neighbor Classification in PCA Space",
    fontsize=14
)

figure.tight_layout()

classification_plot_file = (
    RESULTS_DIRECTORY
    / "test_material_cluster_assignment.png"
)

figure.savefig(
    classification_plot_file,
    dpi=200
)

plt.show()


# ---------------------------------------------------------------------------
# 19. PRINT A SMALL TABLE OF THE RESULTS
# ---------------------------------------------------------------------------

print("First ten test-material results:")
print(
    test_results_df[
        [
            "Material_ID",
            "Assigned_Cluster",
            "Melting_Temperature_K",
            "KNN_Estimated_Melting_Temperature_K",
            "Absolute_Error_K",
        ]
    ].head(10)
)
print()


# ---------------------------------------------------------------------------
# 20. FINAL OUTPUT SUMMARY
# ---------------------------------------------------------------------------

print("Analysis complete.")
print()

print("Training PCA and cluster table:")
print(training_results_file)
print()

print("Test-material results:")
print(test_results_csv)
print()

print("Unsupervised clustering figure:")
print(unsupervised_plot_file)
print()

print("New-material classification figure:")
print(classification_plot_file)
print()


# ---------------------------------------------------------------------------
# PRACTICE EXERCISES
# ---------------------------------------------------------------------------

# 1. Change NUMBER_OF_CLUSTERS from 4 to 3.
#    How does the unsupervised clustering figure change?
#
# 2. Change NUMBER_OF_CLUSTERS from 4 to 5.
#    Are the additional clusters easy to interpret?
#
# 3. Change K_NEIGHBORS from 5 to 3.
#    Does the estimated melting temperature become better or worse?
#
# 4. Change K_NEIGHBORS from 5 to 9.
#    What happens when more distant neighbors are included?
#
# 5. Look at the right side of the unsupervised figure.
#    Do regions of the PCA space correspond to different melting
#    temperatures?
#
# 6. Remove one feature from FEATURE_COLUMNS and run the analysis again.
#    How does the PCA space change?
#
# 7. Compare:
#
#       KMeans clusters
#
#    with:
#
#       melting temperature
#
#    Does an unsupervised cluster automatically correspond to one exact
#    physical property?
#
# 8. Sort test_results_df by Absolute_Error_K.
#    Which new materials were the hardest to predict?
