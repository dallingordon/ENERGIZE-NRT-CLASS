#!/usr/bin/env python3
"""
ME500 Hands-On Day 1
Problem 2 — Student Template (Optional Advanced Challenge)
Which Simulated Atomic Structure Matches Experiment?

The infrastructure is supplied. Your group must design the representation,
preprocessing, classification, testing, and interpretation workflow.
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

SIM_FILE = DATA_DIR / "simulated_rdfs.csv"
EXP_FILE = DATA_DIR / "experimental_rdfs.csv"
GRID_FILE = DATA_DIR / "rdf_grid.csv"
RANDOM_SEED = 801
REGENERATE_DATA = False


def save_figure(fig, filename: str) -> Path:
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {path}")
    return path


def save_model(model, filename="structure_classifier.joblib") -> Path:
    path = MODEL_DIR / filename
    joblib.dump(model, path)
    print(f"Saved model: {path}")
    return path


def load_model(filename="structure_classifier.joblib"):
    path = MODEL_DIR / filename
    if not path.exists():
        raise FileNotFoundError(path)
    return joblib.load(path)


def rdf_columns(frame: pd.DataFrame):
    """Return RDF feature columns in their original stored order."""
    return [c for c in frame.columns if c.startswith("g_r_")]


def frame_to_numpy(frame: pd.DataFrame, columns) -> np.ndarray:
    """Optional helper so DataFrame -> NumPy syntax is not a bottleneck."""
    return frame.loc[:, list(columns)].to_numpy(dtype=float)


def load_r_grid() -> np.ndarray:
    return pd.read_csv(GRID_FILE)["r_A"].to_numpy(dtype=float)


def plot_rdf_set(r, curves, labels=None, title="RDF curves", max_curves=12):
    """Generic plotting helper; it does not choose your ML representation."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    n = min(len(curves), max_curves)
    for i in range(n):
        label = None if labels is None else str(labels[i])
        ax.plot(r, curves[i], alpha=0.75, label=label)
    ax.set_xlabel("r (Å)")
    ax.set_ylabel("g(r)")
    ax.set_title(title)
    if labels is not None and n <= 12:
        # Avoid repeated legend labels where possible.
        handles, text = ax.get_legend_handles_labels()
        unique = dict(zip(text, handles))
        ax.legend(unique.values(), unique.keys(), fontsize=8)
    fig.tight_layout()
    return fig


def save_prediction_table(sample_ids, predicted_labels, filename="experimental_predictions.csv") -> Path:
    out = pd.DataFrame({
        "sample_id": np.asarray(sample_ids),
        "predicted_state": np.asarray(predicted_labels),
    })
    path = RESULTS_DIR / filename
    out.to_csv(path, index=False)
    print(f"Saved predictions: {path}")
    return path


# -----------------------------------------------------------------------------
# 1. SYNTHETIC DATA GENERATION — PROVIDED; DO NOT MODIFY FOR THE EXERCISE
# -----------------------------------------------------------------------------
def _gaussian(r, center, width, amplitude):
    return amplitude * np.exp(-0.5 * ((r - center) / width) ** 2)


def _base_rdf(r, state_index, rng, experimental=False):
    """Generate one synthetic RDF-like curve for classroom use."""
    # Small sample-to-sample perturbations mimic different atomistic snapshots.
    shift_scale = 0.025 if not experimental else 0.075
    width_scale = rng.uniform(0.92, 1.12) if not experimental else rng.uniform(1.15, 1.48)
    noise_scale = 0.045 if not experimental else 0.085
    global_shift = rng.normal(0, shift_scale)

    g = np.ones_like(r)

    if state_index == 0:  # crystal-like
        peaks = [(2.45, 0.11, 2.7), (3.48, 0.15, 1.7), (4.26, 0.18, 1.4), (4.92, 0.21, 1.0), (5.55, 0.24, 0.72), (6.15, 0.28, 0.52)]
    elif state_index == 1:  # defect-rich crystal-like
        peaks = [(2.47, 0.16, 2.35), (3.50, 0.21, 1.38), (4.30, 0.25, 1.08), (4.98, 0.29, 0.75), (5.62, 0.33, 0.46)]
    elif state_index == 2:  # amorphous-like
        peaks = [(2.52, 0.23, 1.85), (4.18, 0.43, 0.72), (5.80, 0.62, 0.32)]
    else:  # liquid-like
        peaks = [(2.60, 0.31, 1.45), (4.65, 0.62, 0.43)]

    for center, width, amp in peaks:
        local_shift = global_shift + rng.normal(0, shift_scale * 0.35)
        local_width = width * width_scale * rng.uniform(0.94, 1.08)
        local_amp = amp * rng.uniform(0.90, 1.10)
        g += _gaussian(r, center + local_shift, local_width, local_amp)

    # Suppress unphysical short-distance probability.
    g *= 1.0 / (1.0 + np.exp(-(r - 2.05) / 0.08))

    if state_index >= 2:
        g += 0.11 * np.exp(-0.55 * (r - 2.2)) * np.sin(5.2 * (r - 2.2))

    if experimental:
        # Experimental-like domain shift: weak slope/background and intensity scaling.
        g = rng.uniform(0.90, 1.10) * g + rng.normal(0, 0.025) * (r - r.mean())

    g += rng.normal(0, noise_scale, len(r))
    return np.clip(g, 0.0, None)


def generate_rdf_datasets(seed=RANDOM_SEED, n_per_state=70, n_experimental=12):
    rng = np.random.default_rng(seed)
    r = np.linspace(1.6, 7.8, 110)
    state_names = np.array(["crystal", "defect_rich_crystal", "amorphous", "liquid"])
    feature_names = [f"g_r_{value:.3f}" for value in r]

    sim_rows = []
    counter = 0
    for state_index, state_name in enumerate(state_names):
        for _ in range(n_per_state):
            curve = _base_rdf(r, state_index, rng, experimental=False)
            row = {"sample_id": f"SIM_{counter:04d}", "state": state_name}
            row.update(dict(zip(feature_names, curve)))
            sim_rows.append(row)
            counter += 1

    # Experimental states are intentionally not written to the student dataset.
    # Treat these as unknown measurements during the activity.
    experimental_state_indices = rng.integers(0, len(state_names), size=n_experimental)
    exp_rows = []
    for i, state_index in enumerate(experimental_state_indices):
        curve = _base_rdf(r, int(state_index), rng, experimental=True)
        row = {"sample_id": f"EXP_{i:03d}"}
        row.update(dict(zip(feature_names, curve)))
        exp_rows.append(row)

    pd.DataFrame(sim_rows).to_csv(SIM_FILE, index=False)
    pd.DataFrame(exp_rows).to_csv(EXP_FILE, index=False)
    pd.DataFrame({"r_A": r, "rdf_column": feature_names}).to_csv(GRID_FILE, index=False)

    metadata = {
        "seed": seed,
        "simulated_samples_per_state": n_per_state,
        "experimental_samples": n_experimental,
        "candidate_states": state_names.tolist(),
        "note": "Experimental labels are intentionally omitted from the student data file.",
    }
    (DATA_DIR / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Generated: {SIM_FILE}")
    print(f"Generated: {EXP_FILE}")
    print(f"Generated: {GRID_FILE}")


def load_datasets():
    if REGENERATE_DATA or not (SIM_FILE.exists() and EXP_FILE.exists() and GRID_FILE.exists()):
        generate_rdf_datasets()
    return pd.read_csv(SIM_FILE), pd.read_csv(EXP_FILE), load_r_grid()


# -----------------------------------------------------------------------------
# 2. STUDENT WORK AREA
# -----------------------------------------------------------------------------
def main():
    sim_df, exp_df, r = load_datasets()

    print("\n" + "=" * 78)
    print("PROBLEM 2: WHICH SIMULATED ATOMIC STRUCTURE MATCHES EXPERIMENT?")
    print("=" * 78)
    print(f"Simulated samples:    {len(sim_df)}")
    print(f"Experimental samples: {len(exp_df)}")
    print(f"RDF points per curve: {len(r)}")
    print("Candidate simulated states:")
    print(sim_df["state"].value_counts().to_string())

    # A basic raw-data plot is allowed infrastructure: it helps everyone inspect
    # the scientific data without solving the ML problem for them.
    cols = rdf_columns(sim_df)
    preview_curves = frame_to_numpy(sim_df.iloc[:12], cols)
    preview_labels = sim_df.iloc[:12]["state"].to_numpy()
    fig = plot_rdf_set(r, preview_curves, preview_labels, title="Example simulated RDFs")
    save_figure(fig, "raw_simulated_rdf_preview.png")

    # -------------------------------------------------------------------------
    # YOUR GROUP'S WORK STARTS HERE
    # -------------------------------------------------------------------------
    # The simulation data have labels. The experimental-like measurements do not.
    # Before predicting experiment, first convince yourselves that your method can
    # distinguish held-out SIMULATED structures that were not used to fit the model.
    #
    # Questions to discuss:
    #   1. What part of each row is the actual structural fingerprint?
    #   2. Do the raw curves require any normalization/transformation before a
    #      model compares them?
    #   3. Is using all RDF points directly sensible, or would a lower-dimensional
    #      representation be useful?
    #   4. How will you create an honest simulation train/test split?
    #   5. What classifier is appropriate and interpretable for this dataset size?
    #   6. After testing on simulation, how will you apply the SAME preprocessing
    #      to experimental data?
    #   7. Can you visualize simulated and experimental samples together in a way
    #      that helps assess whether the experimental points lie inside the domain
    #      represented by simulation?
    #
    # Helpers above handle file paths, array conversion, RDF plotting, model
    # serialization, and prediction-table output. Representation + ML choices are yours.

    # -------------------------------------------------------------------------
    # STEP A: Construct X and y from the simulated data.
    # -------------------------------------------------------------------------


    # -------------------------------------------------------------------------
    # STEP B: Decide on preprocessing / representation.
    # -------------------------------------------------------------------------


    # -------------------------------------------------------------------------
    # STEP C: Split simulated data, train a classifier, and evaluate it honestly.
    # -------------------------------------------------------------------------


    # -------------------------------------------------------------------------
    # STEP D: Apply the same fitted workflow to the experimental samples.
    # -------------------------------------------------------------------------


    # -------------------------------------------------------------------------
    # STEP E: Save predictions/model and make a visualization that supports your
    # scientific interpretation.
    # -------------------------------------------------------------------------


    print("\nTemplate setup is complete.")
    print(f"Figures will be saved to: {FIGURE_DIR}")
    print(f"Models will be saved to:  {MODEL_DIR}")
    print(f"Results will be saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
