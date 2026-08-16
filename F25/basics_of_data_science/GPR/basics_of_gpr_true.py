import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic

# ---------------------------
# Step 1: Generate synthetic training data
# ---------------------------
np.random.seed(42)

# Training data: sin curve with tunable Gaussian noise
X_train = np.linspace(0, 10, 20).reshape(-1, 1)
y_train_true = np.sin(X_train).ravel()
noise_level = 0.2
y_train = y_train_true + noise_level * np.random.randn(len(X_train))

# Test data: completely different distribution (cos curve + higher noise)
X_test = np.linspace(0, 10, 30).reshape(-1, 1)
y_test_true = np.cos(X_test).ravel()
test_noise_level = 0.01
y_test = y_test_true + test_noise_level * np.random.randn(len(X_test))

# Prediction grid for smoother plotting
X_plot = np.linspace(0, 10, 200).reshape(-1, 1)

# ---------------------------
# Step 2: Define kernels and GPR models
# ---------------------------
kernels = [
    RBF(length_scale=1.0),
    Matern(length_scale=1.0, nu=1.5),
    RationalQuadratic(length_scale=1.0, alpha=1.0)
]

kernel_names = ["RBF", "Matern (ν=1.5)", "RationalQuadratic"]

models = [
    GaussianProcessRegressor(kernel=k, alpha=noise_level ** 2, normalize_y=True, random_state=42)
    for k in kernels
]

# ---------------------------
# Step 3: Fit models and make predictions
# ---------------------------
train_preds, train_stds = [], []
test_preds, test_stds = [], []

for model in models:
    # Fit on training data
    model.fit(X_train, y_train)

    # Predict on dense grid (for visualization of training)
    y_pred_train, y_std_train = model.predict(X_plot, return_std=True)
    train_preds.append(y_pred_train)
    train_stds.append(y_std_train)

    # Predict on test data (directly at test points)
    y_pred_test, y_std_test = model.predict(X_test, return_std=True)
    test_preds.append(y_pred_test)
    test_stds.append(y_std_test)

# ---------------------------
# Step 4: Plot results
# ---------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)

# Training row
for i, ax in enumerate(axes[0, :]):
    ax.plot(X_plot, np.sin(X_plot), "k--", label="True function (sin)")
    ax.scatter(X_train, y_train, c="r", label="Train data")

    # Mean prediction with uncertainty band
    ax.plot(X_plot, train_preds[i], "b", label="GPR mean")
    ax.fill_between(
        X_plot.ravel(),
        train_preds[i] - 1.96 * train_stds[i],
        train_preds[i] + 1.96 * train_stds[i],
        alpha=0.2,
        color="blue",
    )
    # Error bars at the dense grid points
    ax.errorbar(
        X_plot.ravel(),
        train_preds[i],
        yerr=1.96 * train_stds[i],
        fmt="none",
        ecolor="blue",
        alpha=0.3,
        capsize=2,
    )

    ax.set_title(f"Training: {kernel_names[i]}")
    ax.legend(loc="upper right")

# Testing row
for i, ax in enumerate(axes[1, :]):
    ax.plot(X_test, y_test_true, "k--", label="True function (cos)")
    ax.scatter(X_test, y_test, c="r", label="Test data")

    # Mean prediction
    ax.plot(X_test, test_preds[i], "b", label="GPR mean")
    # Error bars explicitly on test predictions
    ax.errorbar(
        X_test.ravel(),
        test_preds[i],
        yerr=1.96 * test_stds[i],
        fmt="o",
        color="blue",
        ecolor="blue",
        alpha=0.7,
        capsize=3,
        label="Prediction ±1.96σ"
    )

    ax.set_title(f"Testing: {kernel_names[i]}")
    ax.legend(loc="upper right")

plt.tight_layout()
plt.show()
