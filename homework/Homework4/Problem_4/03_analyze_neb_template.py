#!/usr/bin/env python3
"""
Question 4C: Analyze the NEB energy profile.

STUDENT TASKS
-------------
1. Convert the raw image energies into energies relative to the initial state.
2. Determine the saddle-point image and migration activation energy Ea.
3. Plot relative energy versus reaction coordinate.
4. Save the barrier to results/neb_results.json.

The KMC script will read YOUR migration barrier from that JSON file.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
INPUT_FILE = RESULTS_DIR / "neb_energies.csv"

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        "results/neb_energies.csv was not found. Run 02_run_neb.py first."
    )

df = pd.read_csv(INPUT_FILE)
reaction_coordinate = df["reaction_coordinate"].to_numpy(dtype=float)
energies_eV = df["energy_eV"].to_numpy(dtype=float)


# ============================================================================
# TODO 1 — CALCULATE RELATIVE ENERGIES
# ============================================================================
# Make the first NEB image the zero of energy:
#
#     relative_energy_eV = E_i - E_initial
#
# Store one relative energy for every NEB image.
# ============================================================================
relative_energy_eV = energies_eV - energies_eV[0]


# ============================================================================
# TODO 2 — FIND THE SADDLE AND ACTIVATION ENERGY
# ============================================================================
# Determine:
#   saddle_image_index  = index of the maximum-energy image
#   activation_energy_eV = E_saddle - E_initial
#
# Because relative_energy_eV uses the initial state as zero, the activation
# energy is also the maximum relative energy along the band.
# ============================================================================
saddle_image_index = np.argmax(relative_energy_eV)
activation_energy_eV = relative_energy_eV[saddle_image_index]


if relative_energy_eV is None:
    raise NotImplementedError("TODO 1: calculate the relative NEB energies.")
if saddle_image_index is None or activation_energy_eV is None:
    raise NotImplementedError("TODO 2: determine the saddle image and Ea.")

relative_energy_eV = np.asarray(relative_energy_eV, dtype=float)
saddle_image_index = int(saddle_image_index)
activation_energy_eV = float(activation_energy_eV)

if len(relative_energy_eV) != len(energies_eV):
    raise RuntimeError("relative_energy_eV must contain one value per NEB image.")
if not (0 <= saddle_image_index < len(df)):
    raise RuntimeError("saddle_image_index is outside the NEB image range.")
if activation_energy_eV <= 0:
    raise RuntimeError("The migration activation energy should be positive.")


# ============================================================================
# TODO 3 — MAKE THE NEB ENERGY-PROFILE PLOT
# ============================================================================
# Plot:
#   x-axis: reaction coordinate
#   y-axis: relative energy (eV)
#
# Requirements:
#   * plot all NEB images as points connected by a line
#   * visually identify the saddle-point image
#   * label axes with units
#   * save the figure as:
#         results/al_vacancy_neb.png
# ============================================================================

fig, ax = plt.subplots(figsize=(6, 4))

# all images, points connected by a line
ax.plot(reaction_coordinate, relative_energy_eV, "o-", color="tab:blue", label="NEB images")

# highlight the saddle image
ax.plot(reaction_coordinate[saddle_image_index], relative_energy_eV[saddle_image_index],
        "*", color="tab:red", markersize=16,
        label=f"Saddle (image {saddle_image_index})")
ax.annotate(f"$E_a$ = {activation_energy_eV:.3f} eV",
            xy=(reaction_coordinate[saddle_image_index], relative_energy_eV[saddle_image_index]),
            xytext=(0, 10), textcoords="offset points", ha="center")

ax.set_ylim(-0.02, activation_energy_eV * 1.2)
ax.set_xlabel("Normalized reaction coordinate")
ax.set_ylabel("Energy relative to initial state (eV)")
ax.set_title("Al monovacancy migration (CI-NEB, EAM)")
ax.legend()
fig.tight_layout()
fig.savefig(RESULTS_DIR / "al_vacancy_neb.png", dpi=200)
plt.close(fig)

plot_file = RESULTS_DIR / "al_vacancy_neb.png"
if not plot_file.exists():
    raise FileNotFoundError(
        "TODO 3: save your NEB plot as results/al_vacancy_neb.png"
    )

results = {
    "migration_barrier_eV": activation_energy_eV,
    "saddle_image_index": saddle_image_index,
    "initial_energy_eV": float(energies_eV[0]),
    "saddle_energy_eV": float(energies_eV[saddle_image_index]),
    "final_energy_eV": float(energies_eV[-1]),
    "endpoint_energy_difference_eV": float(energies_eV[-1] - energies_eV[0]),
}

with open(RESULTS_DIR / "neb_results.json", "w") as f:
    json.dump(results, f, indent=2)

summary = df.copy()
summary["relative_energy_eV"] = relative_energy_eV
summary.to_csv(RESULTS_DIR / "neb_energy_profile.csv", index=False)

print("\nNEB ANALYSIS")
print("============")
print(f"Saddle image:               {saddle_image_index}")
print(f"NEB migration barrier Ea:   {activation_energy_eV:.6f} eV")
print(f"Endpoint energy difference: {energies_eV[-1]-energies_eV[0]:+.6e} eV")
print("\nCreated:")
print("  results/al_vacancy_neb.png")
print("  results/neb_results.json")
print("  results/neb_energy_profile.csv")
print("\nThe KMC script will use migration_barrier_eV from neb_results.json.")
