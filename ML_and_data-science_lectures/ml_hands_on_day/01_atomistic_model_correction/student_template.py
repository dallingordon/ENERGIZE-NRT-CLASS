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
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
