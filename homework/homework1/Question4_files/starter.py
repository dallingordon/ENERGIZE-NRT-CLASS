"""ME500 Homework 1, Problem 1 starter code.

Complete the TODO sections. You may reorganize the script if you prefer, but your
final program should perform every step requested in the problem statement.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "data" / "materials_regression.csv"

# -----------------------------------------------------------------------------
# 1. Load the data
# -----------------------------------------------------------------------------
data = pd.read_csv(DATA_FILE)

FEATURES = [
    "avg_atomic_number",
    "avg_electronegativity",
    "density_g_cm3",
    "cohesive_energy_eV_atom",
    "atomic_volume_A3_atom",
    "packing_fraction",
]
TARGET = "bulk_modulus_GPa"

# TODO: create X and y

# -----------------------------------------------------------------------------
# 2. Split the data: 80% training, 20% testing; random_state=42
# -----------------------------------------------------------------------------
# TODO

# -----------------------------------------------------------------------------
# 3. Fit LinearRegression using only the training data
# -----------------------------------------------------------------------------
# TODO

# -----------------------------------------------------------------------------
# 4. Make train/test predictions and calculate MAE and R^2
# -----------------------------------------------------------------------------
# TODO

# -----------------------------------------------------------------------------
# 5. Predicted-versus-actual plot for the test data
# -----------------------------------------------------------------------------
# TODO

plt.tight_layout()
plt.savefig(HERE / "problem1_predicted_vs_actual.png", dpi=200)
plt.close()
