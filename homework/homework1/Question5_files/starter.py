"""ME500 Homework 1, Problem 2 starter code."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score

HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "data" / "materials_scaling_pca.csv"

data = pd.read_csv(DATA_FILE)

FEATURES = [
    "avg_atomic_number",
    "density_g_cm3",
    "band_gap_eV",
    "defect_fraction",
    "porosity_percent",
    "grain_size_nm",
    "mean_bond_strength_eV",
]
TARGET = "thermal_conductivity_W_mK"

# TODO 1: Create X and y and make one 80/20 train/test split.

# TODO 2: Train KNN (n_neighbors=7) on raw features and evaluate test MAE/R^2.

# TODO 3: Fit StandardScaler ONLY to X_train, transform train and test,
#         train the same KNN model, and evaluate it.

# TODO 4: PCA on raw X for visualization.

# TODO 5: PCA on standardized X for visualization.
# Note: for this visualization-only PCA, it is acceptable to fit a new scaler to
# the complete X because this PCA is not being used to estimate test performance.

# TODO 6: Plot PC1 versus PC2 and color by data["material_class"].
