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
print("sup")
#breakpoint() works with the conda env from class
# TODO: create X and y
X = data[FEATURES]
y = data[TARGET]
#print(X.head(10), y.head(10))
#breakpoint()
# -----------------------------------------------------------------------------
# 2. Split the data: 80% training, 20% testing; random_state=42
# -----------------------------------------------------------------------------

res = train_test_split(X, y, test_size=0.2, random_state=42)
#print([x.shape() for x in res])
X_train, X_test, y_train, y_test = res

#breakpoint()

# -----------------------------------------------------------------------------
# 3. Fit LinearRegression using only the training data
# -----------------------------------------------------------------------------

model = LinearRegression()
model.fit(X_train, y_train)
#print(dir(model))
#breakpoint()

# -----------------------------------------------------------------------------
# 4. Make train/test predictions and calculate MAE and R^2
# -----------------------------------------------------------------------------
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)
#print(train_pred.shape) #208, 
#print(test_pred.shape) #52,
#breakpoint()
train_metrics = (mean_absolute_error(train_pred, y_train),r2_score(y_train, train_pred))
#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html true value, pred value
#train_metrics_s = (mean_absolute_error( y_train, train_pred,),r2_score(y_train, train_pred )) mae is identical, r2 isnt.....
test_metrics = (mean_absolute_error(test_pred, y_test),r2_score( y_test, test_pred))

# -----------------------------------------------------------------------------
# 5. Predicted-versus-actual plot for the test data
# -----------------------------------------------------------------------------

#heavily relied on the ML_and_data-science_lectures/ml_workflow_design/model_testing/04_test_mof_neural_network.py plotting (#9, line ~246)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

train_mae = mean_absolute_error(y_train, train_pred)
train_r2 = r2_score(y_train, train_pred)
test_mae = mean_absolute_error(y_test, test_pred)
test_r2 = r2_score(y_test, test_pred)

# Training data
axes[0].scatter(y_train, train_pred, s=20)
lims = [min(y_train.min(), train_pred.min()), max(y_train.max(), train_pred.max())]
axes[0].plot(lims, lims, "k--")
axes[0].set_xlabel("Actual Bulk Modulus (GPa)")
axes[0].set_ylabel("Predicted Bulk Modulus (GPa)")
axes[0].set_title(f"Train\nMAE: {train_mae:.2f} GPa, R²: {train_r2:.3f}")

# Test data
axes[1].scatter(y_test, test_pred, s=20)
lims = [min(y_test.min(), test_pred.min()), max(y_test.max(), test_pred.max())]
axes[1].plot(lims, lims, "k--")
axes[1].set_xlabel("Actual Bulk Modulus (GPa)")
axes[1].set_ylabel("Predicted Bulk Modulus (GPa)")
axes[1].set_title(f"Test\nMAE: {test_mae:.2f} GPa, R²: {test_r2:.3f}")

plt.suptitle("Predicted vs. Actual Bulk Modulus")


plt.tight_layout()
plt.savefig(HERE / "problem1_predicted_vs_actual.png", dpi=200)
plt.close()
