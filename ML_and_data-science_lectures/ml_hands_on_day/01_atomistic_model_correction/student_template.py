#!/usr/bin/env python3
"""
ME500 Hands-On Day 1
Problem 1 — Student Template
Correcting an Imperfect Atomistic Model

This script supplies the infrastructure needed for the exercise, but it does
NOT supply the ML solution.

Already handled for you:
    * reproducible synthetic atomistic + experimental data generation
    * safe file/directory management independent of your working directory
    * automatic regeneration if the instructor changes the dataset settings
    * loading the generated CSV file
    * convenience helpers for NumPy conversion
    * figure saving
    * model saving/loading
    * saving prediction tables

Your group must design:
    * data formatting for the ML problem
    * feature and target choices
    * preprocessing
    * train/test strategy
    * a non-ML baseline
    * model choice, training, and prediction
    * quantitative testing
    * scientific interpretation

You should be able to run this file untouched once. It will create the dataset,
print a short summary, and show you where your group's work begins.
"""

from pathlib import Path
from datetime import datetime
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy import stats

# -----------------------------------------------------------------------------
# 0. FILE MANAGEMENT — PROVIDED INFRASTRUCTURE
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
FIGURE_DIR = SCRIPT_DIR / "figures"
MODEL_DIR = SCRIPT_DIR / "models"
RESULTS_DIR = SCRIPT_DIR / "results"

for directory in (DATA_DIR, FIGURE_DIR, MODEL_DIR, RESULTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

DATA_FILE = DATA_DIR / "atomistic_vs_experiment.csv"
METADATA_FILE = DATA_DIR / "dataset_metadata.json"

# -----------------------------------------------------------------------------
# 1. INSTRUCTOR-CONTROLLED DATA SETTINGS — PROVIDED
# -----------------------------------------------------------------------------
# These settings only control how the synthetic classroom dataset is generated.
# They are NOT part of the ML challenge, so your group does not need to modify
# them. If the instructor changes them, the script automatically regenerates the
# CSV the next time it is run.
RANDOM_SEED = 500
N_MATERIALS = 320
SYSTEMATIC_DISCREPANCY_STRENGTH = 1.50
RANDOM_SIM_TO_EXP_NOISE_GPA = 6.0
REGENERATE_DATA = False
AUTO_REGENERATE_IF_SETTINGS_CHANGE = True


def save_figure(fig, filename: str) -> Path:
    """Save a matplotlib figure without needing to manage paths manually."""
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {path}")
    return path


def save_model(model, filename: str = "trained_model.joblib") -> Path:
    """Save any scikit-learn compatible model or pipeline."""
    path = MODEL_DIR / filename
    joblib.dump(model, path)
    print(f"Saved model: {path}")
    return path


def load_model(filename: str = "trained_model.joblib"):
    """Reload a model previously saved with save_model()."""
    path = MODEL_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"No saved model was found at {path}")
    return joblib.load(path)


def save_prediction_table(ids, y_true, y_pred, filename="test_predictions.csv") -> Path:
    """Convenience function for writing a prediction table."""
    output = pd.DataFrame({
        "material_id": np.asarray(ids),
        "y_true": np.asarray(y_true, dtype=float),
        "y_pred": np.asarray(y_pred, dtype=float),
    })
    path = RESULTS_DIR / filename
    output.to_csv(path, index=False)
    print(f"Saved predictions: {path}")
    return path


def frame_to_numpy(frame: pd.DataFrame, columns) -> np.ndarray:
    """Optional helper so DataFrame -> NumPy syntax is not a bottleneck."""
    return frame.loc[:, list(columns)].to_numpy(dtype=float)


def column_to_numpy(frame: pd.DataFrame, column: str) -> np.ndarray:
    """Optional helper for converting one DataFrame column to a 1D array."""
    return frame[column].to_numpy(dtype=float)


def generation_settings() -> dict:
    """Return the settings that uniquely determine the classroom dataset."""
    return {
        "seed": int(RANDOM_SEED),
        "n_materials": int(N_MATERIALS),
        "systematic_discrepancy_strength": float(SYSTEMATIC_DISCREPANCY_STRENGTH),
        "random_sim_to_exp_noise_GPa": float(RANDOM_SIM_TO_EXP_NOISE_GPA),
    }


def saved_settings_match_current() -> bool:
    """Check whether an existing CSV was generated with the current settings."""
    if not DATA_FILE.exists() or not METADATA_FILE.exists():
        return False

    try:
        metadata = json.loads(METADATA_FILE.read_text())
        return metadata.get("generation_settings") == generation_settings()
    except (OSError, json.JSONDecodeError):
        return False


# -----------------------------------------------------------------------------
# 2. SYNTHETIC DATA GENERATION — PROVIDED; DO NOT MODIFY FOR THE EXERCISE
# -----------------------------------------------------------------------------
def generate_atomistic_dataset(path: Path) -> None:
    """
    Create a deterministic synthetic dataset that mimics an imperfect atomistic
    model compared with experiment.

    Treat the resulting CSV exactly as you would treat supplied simulation and
    experimental data in a real project. The equations below exist only so the
    exercise can be completely self-contained.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    n_materials = N_MATERIALS

    # Hidden variables used only to make the synthetic descriptors physically
    # correlated. They are intentionally NOT included in the output CSV.
    bonding_strength = rng.uniform(0.15, 1.0, n_materials)
    packing = rng.uniform(0.20, 1.0, n_materials)
    directionality = rng.uniform(0.0, 1.0, n_materials)

    cohesive_energy = 1.2 + 5.8 * bonding_strength + rng.normal(0, 0.18, n_materials)

    atomic_volume = (
        25.0
        - 9.0 * packing
        - 2.0 * bonding_strength
        + rng.normal(0, 0.7, n_materials)
    )
    atomic_volume = np.clip(atomic_volume, 8.5, 25.0)

    coordination_number = np.clip(
        4.0 + 8.0 * packing + rng.normal(0, 0.45, n_materials),
        3.0,
        12.5,
    )

    mean_bond_length = (
        3.05
        - 0.48 * packing
        - 0.22 * bonding_strength
        + rng.normal(0, 0.035, n_materials)
    )

    bulk_modulus = (
        18.0
        + 34.0 * cohesive_energy
        + 22.0 * packing
        - 2.0 * atomic_volume
        + rng.normal(0, 8.0, n_materials)
    )
    bulk_modulus = np.clip(bulk_modulus, 15.0, 300.0)

    shear_modulus = (
        0.47 * bulk_modulus
        + 22.0 * directionality
        - 7.0
        + rng.normal(0, 6.0, n_materials)
    )
    shear_modulus = np.clip(shear_modulus, 7.0, 190.0)

    young_sim = 9.0 * bulk_modulus * shear_modulus / (3.0 * bulk_modulus + shear_modulus)
    young_sim += rng.normal(0, 5.0, n_materials)

    # The atomistic model has a systematic, descriptor-dependent discrepancy
    # relative to experiment, plus an additional random component.
    base_systematic_correction = (
        34.0
        - 0.16 * young_sim
        + 10.0 * (cohesive_energy - cohesive_energy.mean())
        - 3.2 * (atomic_volume - atomic_volume.mean())
        + 0.11 * (bulk_modulus - bulk_modulus.mean())
        + 2.8 * (coordination_number - coordination_number.mean())
        + 9.0 * np.sin(1.7 * mean_bond_length)
    )

    systematic_correction = (
        SYSTEMATIC_DISCREPANCY_STRENGTH * base_systematic_correction
    )

    random_discrepancy = rng.normal(
        0.0,
        RANDOM_SIM_TO_EXP_NOISE_GPA,
        n_materials,
    )

    young_exp = np.clip(
        young_sim + systematic_correction + random_discrepancy,
        8.0,
        None,
    )

    df = pd.DataFrame({
        "material_id": [f"MAT_{i:04d}" for i in range(n_materials)],
        "cohesive_energy_eV_per_atom": cohesive_energy,
        "atomic_volume_A3_per_atom": atomic_volume,
        "coordination_number": coordination_number,
        "mean_bond_length_A": mean_bond_length,
        "bulk_modulus_sim_GPa": bulk_modulus,
        "shear_modulus_sim_GPa": shear_modulus,
        "youngs_modulus_sim_GPa": young_sim,
        "youngs_modulus_exp_GPa": young_exp,
    })
    df.to_csv(path, index=False)

    metadata = {
        "description": (
            "Synthetic atomistic descriptors plus simulated and experimental "
            "Young's modulus for the ME500 ML hands-on exercise."
        ),
        "generation_settings": generation_settings(),
        "units": {
            "cohesive_energy_eV_per_atom": "eV/atom",
            "atomic_volume_A3_per_atom": "angstrom^3/atom",
            "coordination_number": "dimensionless",
            "mean_bond_length_A": "angstrom",
            "bulk_modulus_sim_GPa": "GPa",
            "shear_modulus_sim_GPa": "GPa",
            "youngs_modulus_sim_GPa": "GPa",
            "youngs_modulus_exp_GPa": "GPa",
        },
    }
    METADATA_FILE.write_text(json.dumps(metadata, indent=2))
    print(f"Generated dataset: {path}")


def load_dataset() -> pd.DataFrame:
    should_regenerate = REGENERATE_DATA or not DATA_FILE.exists()

    if AUTO_REGENERATE_IF_SETTINGS_CHANGE and not saved_settings_match_current():
        should_regenerate = True

    if should_regenerate:
        generate_atomistic_dataset(DATA_FILE)
    else:
        print(f"Reusing existing dataset: {DATA_FILE}")

    return pd.read_csv(DATA_FILE)


# -----------------------------------------------------------------------------
# 3. STUDENT WORK AREA
# -----------------------------------------------------------------------------
def main() -> None:
    df = load_dataset()

    print("\n" + "=" * 78)
    print("PROBLEM 1: CORRECTING AN IMPERFECT ATOMISTIC MODEL")
    print("=" * 78)
    print(f"Dataset: {DATA_FILE}")
    print(f"Rows: {len(df)}")
    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")
    print("\nFirst five rows:")
    print(df.head().to_string(index=False))


    # --- Residual target: delta = experiment - simulation (in GPa) ----------
    # Computed per material from its own two values, so doing it before the
    # split leaks nothing. Because it contains the experimental answer, delta
    # is a TARGET -- it must never be used as an input feature.
    TARGET_COL = "youngs_modulus_exp_GPa"
    SIM_COL = "youngs_modulus_sim_GPa"
    DELTA_COL = "delta_youngs_modulus_GPa"
    df[DELTA_COL] = df[TARGET_COL] - df[SIM_COL]

    # --- Histograms of every numeric column (raw units, incl. delta) ---------------------------------
    numeric_cols = df.select_dtypes(include="number").columns
    n_cols = 3
    n_rows = int(np.ceil(len(numeric_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, column in zip(axes, numeric_cols):
        ax.hist(df[column], bins=30, color="steelblue", edgecolor="black")
        ax.set_title(column)
        ax.set_xlabel(column)
        ax.set_ylabel("Count")
    for ax in axes[len(numeric_cols):]:
        ax.set_visible(False)
    fig.tight_layout()
    save_figure(fig, "column_histograms.png")

    # --- Train/test split (done BEFORE any scaling) --------------------------
    TEST_FRACTION = 0.2
    split_rng = np.random.default_rng(RANDOM_SEED)
    shuffled_idx = split_rng.permutation(len(df))
    n_test = int(round(TEST_FRACTION * len(df)))
    df_test = df.iloc[shuffled_idx[:n_test]].reset_index(drop=True)
    df_train = df.iloc[shuffled_idx[n_test:]].reset_index(drop=True)
    print(f"\nTrain rows: {len(df_train)}   Test rows: {len(df_test)}")

    # --- Metric helpers + results report ------------------------------------
    # Every line passed to report() is printed AND collected so it can be
    # saved to a timestamped text file in results/ at the end.
    report_lines = []

    def report(line=""):
        print(line)
        report_lines.append(line)

    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def mae(a, b):
        return float(np.mean(np.abs(a - b)))

    def r2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return float(1 - ss_res / ss_tot)

    # --- Baseline: use the raw simulated modulus as the prediction ----------
    # No ML, no scaling -- this is the number any correction has to beat.
    report("\nBaseline -- simulated Young's modulus as the prediction (GPa):")
    report(f"  {'':10s} {'RMSE':>7s} {'MAE':>7s} {'R^2':>7s} {'mean bias':>10s}")
    for name, frame in [("Train", df_train), ("Test", df_test)]:
        E_exp = frame[TARGET_COL].to_numpy()
        E_sim = frame[SIM_COL].to_numpy()
        report(
            f"  {name:10s} {rmse(E_sim, E_exp):7.2f} {mae(E_sim, E_exp):7.2f} "
            f"{r2(E_exp, E_sim):7.3f} {np.mean(E_sim - E_exp):+10.2f}"
        )
    report("  (mean bias = average of sim - exp; positive means the simulation overpredicts)")

    # --- Min-max scale the input features to [-1, 1] ------------------------
    # min and max come from the TRAINING rows only, then the same numbers are
    # applied to the test rows (and to any new material at inference time).
    #   scaled = 2 * (x - min) / (max - min) - 1
    #   x      = (scaled + 1) / 2 * (max - min) + min
    # Test values can fall slightly outside [-1, 1] if they are more extreme
    # than anything in training -- a useful extrapolation warning.
    # delta is scaled the same way (with its own training min/max) so a model
    # can predict it in scaled units; undo with the inverse formula above.
    # The raw experimental modulus stays unscaled in GPa for evaluation.
    feature_cols = [
        c
        for c in df.select_dtypes(include="number").columns
        if c not in (TARGET_COL, DELTA_COL)
    ]
    scaled_cols = feature_cols + [DELTA_COL]
    df_mus = pd.DataFrame(
        {"min": df_train[scaled_cols].min(), "max": df_train[scaled_cols].max()}
    ).T

    def scale_features(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        span = df_mus.loc["max"] - df_mus.loc["min"]
        out[scaled_cols] = 2 * (frame[scaled_cols] - df_mus.loc["min"]) / span - 1
        return out

    df_train_normalized = scale_features(df_train)
    df_test_normalized = scale_features(df_test)
    print("\nScaling parameters from training set (df_mus):")
    print(df_mus.to_string())
    print("sup bitch")

    # --- Histograms of the scaled training features + delta ----------------
    n_cols = 3
    n_rows = int(np.ceil(len(scaled_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, column in zip(axes, scaled_cols):
        ax.hist(df_train_normalized[column], bins=30, color="steelblue", edgecolor="black")
        ax.set_title(column)
        ax.set_xlabel(f"{column} (scaled to [-1, 1], train)")
        ax.set_ylabel("Count")
    for ax in axes[len(scaled_cols):]:
        ax.set_visible(False)
    fig.tight_layout()
    save_figure(fig, "normalized_histograms.png")

    # --- Correlation plots: each input feature vs delta (exp - sim) ---------
    # Scaled inputs on x, delta (GPa) on y. Pearson r in each title is
    # computed on the TRAINING rows only.
    n_cols = 3
    n_rows = int(np.ceil(len(feature_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, column in zip(axes, feature_cols):
        ax.scatter(df_train_normalized[column], df_train[DELTA_COL], s=12,
                   alpha=0.6, color="steelblue", edgecolor="none", label="Train")
        ax.scatter(df_test_normalized[column], df_test[DELTA_COL], s=12,
                   alpha=0.8, color="darkorange", edgecolor="none", label="Test")
        r_train = np.corrcoef(df_train_normalized[column], df_train[DELTA_COL])[0, 1]
        ax.set_title(f"{column}  (r = {r_train:+.2f})")
        ax.set_xlabel(f"{column} (scaled)")
        ax.axhline(0, color="k", linestyle=":", linewidth=1)
        ax.set_ylabel("delta = exp - sim (GPa)")
    axes[0].legend(loc="best", fontsize=8)
    for ax in axes[len(feature_cols):]:
        ax.set_visible(False)
    fig.tight_layout()
    save_figure(fig, "feature_vs_delta_correlations.png")

    # --- Simulated vs experimental modulus, and delta vs simulated ----------
    # All values on the [-1, 1] scale. The experimental modulus is put on the
    # SAME scale as the simulated modulus (using the sim column's training
    # min/max) so the 1:1 line still means "simulation == experiment".
    def to_sim_scale(values):
        lo, hi = df_mus.loc["min", SIM_COL], df_mus.loc["max", SIM_COL]
        return 2 * (values - lo) / (hi - lo) - 1

    splits = [
        ("Train", df_train, df_train_normalized, "steelblue"),
        ("Test", df_test, df_test_normalized, "darkorange"),
    ]
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5.5))

    for label, raw, scaled, color in splits:
        ax_left.scatter(
            to_sim_scale(raw[TARGET_COL]), scaled[SIM_COL],
            s=18, alpha=0.7, color=color, edgecolor="none", label=label,
        )
    lims = [
        min(ax_left.get_xlim()[0], ax_left.get_ylim()[0]),
        max(ax_left.get_xlim()[1], ax_left.get_ylim()[1]),
    ]
    ax_left.plot(lims, lims, "k:", linewidth=1.5, label="1:1")
    ax_left.set_xlim(lims)
    ax_left.set_ylim(lims)
    ax_left.set_aspect("equal")
    ax_left.set_xlabel("Experimental Young's modulus (scaled)")
    ax_left.set_ylabel("Simulated Young's modulus (scaled)")
    ax_left.set_title("Simulation vs experiment")
    ax_left.legend()

    for label, raw, scaled, color in splits:
        ax_right.scatter(
            scaled[SIM_COL], scaled[DELTA_COL],
            s=18, alpha=0.7, color=color, edgecolor="none", label=label,
        )
    ax_right.set_xlabel("Simulated Young's modulus (scaled)")
    ax_right.set_ylabel("Delta = exp - sim (scaled)")
    ax_right.set_title("Simulation error vs simulated value")
    ax_right.legend()

    fig.tight_layout()
    save_figure(fig, "sim_vs_exp_and_delta.png")

    #now a linear reg:
    # Inputs: the 7 simulation descriptors ONLY -- no experimental modulus and
    # no delta (both contain the experimental answer).
    input_cols = [
        "cohesive_energy_eV_per_atom",
        "atomic_volume_A3_per_atom",
        "coordination_number",
        "mean_bond_length_A",
        "bulk_modulus_sim_GPa",
        "shear_modulus_sim_GPa",
        "youngs_modulus_sim_GPa",
    ]
    output_col = DELTA_COL  # "delta_youngs_modulus_GPa" (scaled to [-1, 1])

    X_train = df_train_normalized[input_cols].to_numpy()
    y_train = df_train_normalized[output_col].to_numpy()
    X_test = df_test_normalized[input_cols].to_numpy()
    y_test = df_test_normalized[output_col].to_numpy()

    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)

    # Predict scaled delta on the held-out test set, un-scale it back to GPa,
    # then add it to the simulated modulus to get the corrected prediction.
    delta_pred_scaled = lin_reg.predict(X_test)
    d_lo, d_hi = df_mus.loc["min", DELTA_COL], df_mus.loc["max", DELTA_COL]
    delta_pred_GPa = (delta_pred_scaled + 1) / 2 * (d_hi - d_lo) + d_lo
    E_sim_test = df_test[SIM_COL].to_numpy()
    E_exp_test = df_test[TARGET_COL].to_numpy()
    E_corrected_test = E_sim_test + delta_pred_GPa

    # Train-set predictions too (for R^2 / overfitting check)
    delta_fit_scaled = lin_reg.predict(X_train)
    E_sim_train = df_train[SIM_COL].to_numpy()
    E_exp_train = df_train[TARGET_COL].to_numpy()
    E_corrected_train = E_sim_train + (delta_fit_scaled + 1) / 2 * (d_hi - d_lo) + d_lo

    # now graph:
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5.5))

    # Left: parity plot on the held-out test set, in GPa
    ax_left.scatter(E_exp_test, E_sim_test, s=22, alpha=0.7, color="gray",
                    edgecolor="none", label="Raw simulation")
    ax_left.scatter(E_exp_test, E_corrected_test, s=22, alpha=0.85, color="crimson",
                    edgecolor="none", label="Sim + linear delta")
    lims = [
        min(ax_left.get_xlim()[0], ax_left.get_ylim()[0]),
        max(ax_left.get_xlim()[1], ax_left.get_ylim()[1]),
    ]
    ax_left.plot(lims, lims, "k:", linewidth=1.5, label="1:1")
    ax_left.set_xlim(lims)
    ax_left.set_ylim(lims)
    ax_left.set_aspect("equal")
    ax_left.set_xlabel("Experimental Young's modulus (GPa)")
    ax_left.set_ylabel("Predicted Young's modulus (GPa)")
    ax_left.set_title("Test set: raw vs ML-corrected simulation")
    ax_left.legend()

    # Right: predicted vs true delta (scaled), train and test
    ax_right.scatter(y_train, delta_fit_scaled, s=18, alpha=0.6, color="steelblue",
                     edgecolor="none", label="Train")
    ax_right.scatter(y_test, delta_pred_scaled, s=22, alpha=0.85, color="darkorange",
                     edgecolor="none", label="Test")
    ax_right.plot([-1.1, 1.1], [-1.1, 1.1], "k:", linewidth=1.5, label="1:1")
    ax_right.set_xlim(-1.1, 1.1)
    ax_right.set_ylim(-1.1, 1.1)
    ax_right.set_aspect("equal")
    ax_right.set_xlabel("True delta (scaled)")
    ax_right.set_ylabel("Predicted delta (scaled)")
    ax_right.set_title("Linear regression on delta")
    ax_right.legend()

    fig.tight_layout()
    save_figure(fig, "linear_delta_results.png")

    # now show coefficients, R^2, etc.
    # Standard errors, t-statistics and p-values for each coefficient (OLS),
    # computed on the TRAINING data the model was fit on.
    #   SE = sqrt(diag( sigma^2 * (X^T X)^-1 )),  sigma^2 = SSR / (n - p - 1)
    #   t  = coef / SE,   p = two-sided p-value from a t distribution
    X_design = np.column_stack([np.ones(len(X_train)), X_train])  # intercept column
    residuals = y_train - delta_fit_scaled
    dof = X_design.shape[0] - X_design.shape[1]
    sigma2 = np.sum(residuals ** 2) / dof
    cov_beta = sigma2 * np.linalg.inv(X_design.T @ X_design)
    coef_all = np.concatenate([[lin_reg.intercept_], lin_reg.coef_])
    se_all = np.sqrt(np.diag(cov_beta))
    t_all = coef_all / se_all
    p_all = 2 * stats.t.sf(np.abs(t_all), dof)

    report("\nLinear regression on delta -- coefficients (scaled units, train fit):")
    # Significance stars by how many standard errors the coefficient is from 0:
    #   *  |t| >= 1   (~1 sigma, p < 0.32)
    #   ** |t| >= 2   (~2 sigma, p < 0.05)
    #   *** |t| >= 3  (~3 sigma, p < 0.003)
    def stars(t_val):
        return "*" * int(min(np.floor(abs(t_val)), 3))

    report(f"  {'term':32s} {'coef':>8s} {'std err':>8s} {'t':>8s} {'p-value':>10s}")
    for name, c, se, t_val, p_val in zip(
        ["intercept"] + input_cols, coef_all, se_all, t_all, p_all
    ):
        report(f"  {name:32s} {c:+8.3f} {se:8.3f} {t_val:+8.2f} {p_val:10.3g} {stars(t_val)}")
    report("  (* |t|>=1 sigma,  ** |t|>=2 sigma,  *** |t|>=3 sigma)")

    report("\nR^2 of the delta fit (scaled delta):")
    report(f"  Train  {r2(y_train, delta_fit_scaled):.3f}")
    report(f"  Test   {r2(y_test, delta_pred_scaled):.3f}")

    report("\nError vs experiment (GPa):")
    report(f"  {'':22s} {'RMSE':>7s} {'MAE':>7s} {'R^2':>7s}")
    for name, E_exp, E_pred in [
        ("Train raw sim", E_exp_train, E_sim_train),
        ("Train sim + delta", E_exp_train, E_corrected_train),
        ("Test raw sim", E_exp_test, E_sim_test),
        ("Test sim + delta", E_exp_test, E_corrected_test),
    ]:
        report(f"  {name:22s} {rmse(E_pred, E_exp):7.2f} {mae(E_pred, E_exp):7.2f} {r2(E_exp, E_pred):7.3f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = RESULTS_DIR / f"linear_delta_results_{timestamp}.txt"
    results_path.write_text("\n".join(report_lines).lstrip("\n") + "\n")
    print(f"\nSaved results: {results_path}")

    #breakpoint()
   
    # -------------------------------------------------------------------------
    # YOUR GROUP'S WORK STARTS HERE
    # -------------------------------------------------------------------------
    # You have an atomistic prediction, additional atomistic descriptors, and an
    # experimentally measured property. Your goal is to decide how ML can be used
    # to make the simulation more useful without leaking the experimental answer.
    #
    # Before coding, discuss:
    #
    #   1. What exactly should the ML model learn?
    #      The experimental value itself is one possibility, but is there a way
    #      to formulate the problem that explicitly uses the existing simulation?
    #
    #   2. Which supplied quantities are legitimate model inputs?
    #      Ask whether each quantity would be available before the experimental
    #      value you are trying to predict is known.
    #
    #   3. Do your selected features need any preprocessing for the algorithm you
    #      choose? Inspect their scales and distributions before deciding.
    #
    #   4. How will you create truly unseen data for final evaluation?
    #
    #   5. What non-ML baseline should your method beat before you claim that ML
    #      improved the atomistic prediction?
    #
    #   6. Which model is sensible for this dataset size and your chosen feature
    #      representation? Complexity is not automatically better.
    #
    #   7. What quantitative metrics and plots will let you compare the raw
    #      simulation against the ML-assisted result fairly?
    #
    #   8. Where does your final model fail, and when would you not trust it?
    #
    # The helper functions above remove routine Python/file-management barriers.
    # You may use them or equivalent code of your own.

    # -------------------------------------------------------------------------
    # STEP A: Explore the supplied data and formulate the ML problem.
    # -------------------------------------------------------------------------


    # -------------------------------------------------------------------------
    # STEP B: Select/format features and target(s), then perform any preprocessing
    # that your chosen algorithm requires.
    # -------------------------------------------------------------------------


    # -------------------------------------------------------------------------
    # STEP C: Create a defensible training/testing strategy and calculate the
    # performance of a non-ML baseline on the same held-out data.
    # -------------------------------------------------------------------------


    # -------------------------------------------------------------------------
    # STEP D: Build and train your ML model using only the training information.
    # -------------------------------------------------------------------------


    # -------------------------------------------------------------------------
    # STEP E: Apply the trained workflow to the held-out data. Calculate metrics
    # and compare the result quantitatively against your baseline.
    # -------------------------------------------------------------------------


    # -------------------------------------------------------------------------
    # STEP F: Create at least one clear scientific figure showing the comparison.
    # Save the final model and prediction table. Identify at least one situation
    # in which you would NOT trust the final ML-assisted prediction.
    # -------------------------------------------------------------------------


    print("\nTemplate setup is complete.")
    print("Your group's ML workflow should be implemented in the marked work area.")
    print(f"Figures will be saved to: {FIGURE_DIR}")
    print(f"Models will be saved to:  {MODEL_DIR}")
    print(f"Results will be saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
