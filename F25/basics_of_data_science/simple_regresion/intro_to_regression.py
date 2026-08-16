import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# --------------------------
# Setup
# --------------------------
np.random.seed(42)
n = 50
X_plot = np.linspace(-5, 5, 200).reshape(-1, 1)

# Helper function to add noise
def add_noise(y, noise_level=0.5):
    return y + np.random.normal(0, noise_level, y.shape)

# --------------------------
# Generate datasets
# --------------------------

# Dataset 1: Linear relationship with noise
X_lin = np.random.uniform(-5, 5, n).reshape(-1, 1)
y_lin_true = 2 * X_lin[:, 0] + 5
y_lin = add_noise(y_lin_true, noise_level=2.0)

# Dataset 2: Tanh relationship with noise
X_tanh = np.random.uniform(-5, 5, n).reshape(-1, 1)
y_tanh_true = 5 * np.tanh(X_tanh[:, 0]) + 2
y_tanh = add_noise(y_tanh_true, noise_level=0.5)

# --------------------------
# Fit models for Dataset 1 (Linear)
# --------------------------







# --------------------------
# Fit models for Dataset 2 (Tanh)
# --------------------------






# --------------------------
# Predictions for plotting
# --------------------------
y_lin_1 = lin_reg_1.predict(X_plot)
y_poly_1 = poly_reg_1.predict(X_plot)
y_ridge_1 = ridge_reg_1.predict(X_plot)
y_true_1 = 2 * X_plot[:, 0] + 5

y_lin_2 = lin_reg_2.predict(X_plot)
y_poly_2 = poly_reg_2.predict(X_plot)
y_ridge_2 = ridge_reg_2.predict(X_plot)
y_true_2 = 5 * np.tanh(X_plot[:, 0]) + 2

# --------------------------
# Plot results
# --------------------------
fig, axes = plt.subplots(1, 2, figsize=(8, 10))

# Plot Dataset 1 (Linear)
axes[0].scatter(X_lin, y_lin, color="black", label="Data (with noise)")
axes[0].plot(X_plot, y_lin_1, label="Linear Regression", color="red")
axes[0].plot(X_plot, y_poly_1, label="Polynomial Regression (deg=2)", color="blue")
axes[0].plot(X_plot, y_ridge_1, label="Ridge Regression (deg=2)", color="green")
axes[0].plot(X_plot, y_true_1, label="True Function", color="orange", linestyle="--")
axes[0].set_title("Dataset 1: Linear Relationship")
axes[0].set_xlabel("X")
axes[0].set_ylabel("y")
axes[0].legend()

# Plot Dataset 2 (Tanh)
axes[1].scatter(X_tanh, y_tanh, color="black", label="Data (with noise)")
axes[1].plot(X_plot, y_lin_2, label="Linear Regression", color="red")
axes[1].plot(X_plot, y_poly_2, label="Polynomial Regression (deg=2)", color="blue")
axes[1].plot(X_plot, y_ridge_2, label="Ridge Regression (deg=2)", color="green")
axes[1].plot(X_plot, y_true_2, label="True Function", color="orange", linestyle="--")
axes[1].set_title("Dataset 2: Tanh Relationship")
axes[1].set_xlabel("X")
axes[1].set_ylabel("y")
axes[1].legend()

plt.tight_layout()
plt.show()
