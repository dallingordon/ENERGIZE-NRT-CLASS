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

###conda activate materials_by_design_local   

# copied from question 4
X = data[FEATURES] #[300 rows x 7 columns]
y = data[TARGET] #Name: thermal_conductivity_W_mK, Length: 300, dtype: float64
res = train_test_split(X, y, test_size=0.2, random_state=42)
#print([x.shape() for x in res])
X_train, X_test, y_train, y_test = res

##breakpoint()
# TODO 2: Train KNN (n_neighbors=7) on raw features and evaluate test MAE/R^2.
#from sklearn.neighbors import KNeighborsRegressor
#what is the command for this? this was a command for claudette:
knn = KNeighborsRegressor(n_neighbors=7)
knn.fit(X_train, y_train) 
pred = knn.predict(X_train) #list len 60, matches y_test
#breakpoint() 
## this is just diagnostic, training set performance:
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(y_train, pred, s=20)
lims = [min(y_train.min(), pred.min()), max(y_train.max(), pred.max())]
ax.plot(lims, lims, "k--")
ax.set_xlabel("Actual Thermal Conductivity (W/m·K)")
ax.set_ylabel("Predicted Thermal Conductivity (W/m·K)")
ax.set_title("KNN (k=7), raw features: Predicted vs. Actual")
plt.tight_layout()
plt.savefig(HERE / "knn_raw_predicted_vs_actual_train.png", dpi=200)
plt.close()
#evaluate test MAE/R^2. print statement here:
test_pred = knn.predict(X_test) #list len 60, matches y_test
raw_test_metrics = (mean_absolute_error(y_test, test_pred),r2_score(y_test, test_pred))
#print(f"Raw KNN test  MAE: {raw_test_metrics[0]:.2f} W/m·K,  R²: {raw_test_metrics[1]:.3f}")

#   Raw KNN test  MAE: 34.52 W/m·K,  R²: 0.333
#breakpoint()

# TODO 3: Fit StandardScaler ONLY to X_train, transform train and test,
#         train the same KNN model, and evaluate it.
scaler = StandardScaler()
scaler.fit(X_train)
X_train_norm = scaler.transform(X_train)
X_test_norm = scaler.transform(X_test)


knn.fit(X_train_norm, y_train) 
pred_norm = knn.predict(X_train_norm) 
test_pred_norm = knn.predict(X_test_norm)
# plot: standardized-feature KNN, test set
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(y_test, test_pred_norm, s=20)
lims = [min(y_test.min(), test_pred_norm.min()), max(y_test.max(), test_pred_norm.max())]
ax.plot(lims, lims, "k--")
ax.set_xlabel("Actual Thermal Conductivity (W/m·K)")
ax.set_ylabel("Predicted Thermal Conductivity (W/m·K)")
ax.set_title("KNN (k=7), standardized features: Predicted vs. Actual (test)")
plt.tight_layout()
plt.savefig(HERE / "knn_scaled_predicted_vs_actual_test.png", dpi=200)
plt.close()
scaled_test_metrics = (mean_absolute_error(y_test, test_pred_norm),r2_score(y_test, test_pred_norm))
#print(f"Normalized KNN test  MAE: {raw_test_metrics[0]:.2f} W/m·K,  R²: {raw_test_metrics[1]:.3f}")
#print the values for the unnormalized and normalized metrics here:
results = pd.DataFrame(
    {
        "Test MAE (W/m·K)": [raw_test_metrics[0], scaled_test_metrics[0]],
        "Test R²": [raw_test_metrics[1], scaled_test_metrics[1]],
    },
    index=["Raw features", "Standardized features"],
)
print("\nKNN (k=7) test performance")
print(results.round(3).to_string())
mae_change = scaled_test_metrics[0] - raw_test_metrics[0]
print(f"MAE change (standardized - raw): {mae_change:+.2f} W/m·K "
      f"({100 * mae_change / raw_test_metrics[0]:+.1f}%)")

# TODO 4: PCA on raw X for visualization.
#PCA visualization goes here:
pca_raw = PCA(n_components=2)
pcs_raw = pca_raw.fit_transform(X)
print("\nRaw PCA explained variance ratio (PC1, PC2):", pca_raw.explained_variance_ratio_.round(4))
# loadings, for part F
print("Raw PCA loadings:")
print(pd.DataFrame(pca_raw.components_.T, index=FEATURES, columns=["PC1", "PC2"]).round(4).to_string())

# TODO 5: PCA on standardized X for visualization.
# Note: for this visualization-only PCA, it is acceptable to fit a new scaler to
# the complete X because this PCA is not being used to estimate test performance.
X_std_all = StandardScaler().fit_transform(X)
pca_std = PCA(n_components=2)
pcs_std = pca_std.fit_transform(X_std_all)
print("\nStandardized PCA explained variance ratio (PC1, PC2):", pca_std.explained_variance_ratio_.round(4))
print("Standardized PCA loadings:")
print(pd.DataFrame(pca_std.components_.T, index=FEATURES, columns=["PC1", "PC2"]).round(4).to_string())

# TODO 6: Plot PC1 versus PC2 and color by data["material_class"].
classes = data["material_class"]
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, pcs, pca, title in [
    (axes[0], pcs_raw, pca_raw, "PCA on raw features"),
    (axes[1], pcs_std, pca_std, "PCA on standardized features"),
]:
    for cls in sorted(classes.unique()):
        mask = (classes == cls).to_numpy()
        ax.scatter(pcs[mask, 0], pcs[mask, 1], s=18, alpha=0.8, label=cls)
    evr = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({100 * evr[0]:.1f}% of variance)")
    ax.set_ylabel(f"PC2 ({100 * evr[1]:.1f}% of variance)")
    ax.set_title(title)
    ax.legend(title="material_class")
plt.suptitle("PC1 vs. PC2, colored by material class")
plt.tight_layout()
plt.savefig(HERE / "pca_raw_vs_standardized.png", dpi=200)
plt.close()


# mean and std of the raw input features
print("\nRaw feature mean and std:")
print(X.agg(["mean", "std"]).T.to_string())

# Raw feature mean and std:
#                              mean         std
# avg_atomic_number       33.715910   13.334020
# density_g_cm3            5.225432    1.715995
# band_gap_eV              1.745801    1.593954
# defect_fraction          0.016721    0.010422
# porosity_percent         7.745445    4.298622
# grain_size_nm          341.130127  418.275062 ##This is a big component of PC1 that is absorbing everything. My guess
# mean_bond_strength_eV    3.783238    0.980081