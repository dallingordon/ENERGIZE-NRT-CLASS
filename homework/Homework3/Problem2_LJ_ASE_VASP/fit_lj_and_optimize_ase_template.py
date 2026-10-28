#!/usr/bin/env python3
"""
Problem 2 starter script
Fit a Lennard-Jones potential to fictitious Al2 data and use it in ASE.

This script already:
  * reads the fictitious training data,
  * fits epsilon and sigma,
  * reports fit quality,
  * plots the fitted LJ curve,
  * creates the ASE LennardJones calculator,
  * writes fitted parameters to JSON,
  * writes final ASE results after optimization.

YOU must:
  1. construct the initial positions of the two Al atoms,
  2. construct the ASE Atoms object,
  3. implement and run the ASE geometry optimization.

The intended starting structure is:
  * Al2
  * 20 A cubic cell
  * initial Al-Al separation = 3.50 A
  * molecule centered well away from the box boundaries
  * non-periodic ASE calculation (pbc=False)
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.io import write

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "al_dimer_fictitious.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CELL_LENGTH_A = 20.0
INITIAL_DISTANCE_A = 3.50
FMAX_TARGET_EV_PER_A = 0.01


def lj_energy(r, epsilon, sigma):
    r = np.asarray(r, dtype=float)
    return 4.0 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)


# =====================================================================
# PART A — FIT THE LJ PARAMETERS
# =====================================================================
data = pd.read_csv(DATA_FILE)
r_data = data["distance_A"].to_numpy()
e_data = data["energy_eV"].to_numpy()

popt, pcov = curve_fit(
    lj_energy,
    r_data,
    e_data,
    p0=(1.0, 2.4),
    bounds=([0.0, 0.5], [10.0, 6.0]),
)

epsilon_fit, sigma_fit = popt
e_pred = lj_energy(r_data, epsilon_fit, sigma_fit)
rmse = np.sqrt(np.mean((e_pred - e_data) ** 2))
r_eq_analytic = (2.0 ** (1.0 / 6.0)) * sigma_fit

print("FITTED LENNARD-JONES PARAMETERS")
print("===============================")
print(f"epsilon = {epsilon_fit:.6f} eV")
print(f"sigma   = {sigma_fit:.6f} A")
print(f"RMSE    = {rmse:.6f} eV")
print(f"Analytical LJ equilibrium distance = {r_eq_analytic:.6f} A")

fit_parameters = {
    "epsilon_eV": float(epsilon_fit),
    "sigma_A": float(sigma_fit),
    "fit_rmse_eV": float(rmse),
    "analytic_equilibrium_distance_A": float(r_eq_analytic),
}
with open(RESULTS_DIR / "lj_fit_parameters.json", "w") as f:
    json.dump(fit_parameters, f, indent=2)

r_plot = np.linspace(r_data.min(), r_data.max(), 500)
e_plot = lj_energy(r_plot, epsilon_fit, sigma_fit)
plt.figure(figsize=(6.5, 4.5))
plt.scatter(r_data, e_data, label="Fictitious reference data")
plt.plot(r_plot, e_plot, label="Fitted LJ potential")
plt.axvline(r_eq_analytic, linestyle="--", label="Analytical LJ minimum")
plt.xlabel("Al-Al distance (A)")
plt.ylabel("Energy (eV)")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "lj_fit.png", dpi=200)
plt.close()

pd.DataFrame({"distance_A": r_plot, "fitted_energy_eV": e_plot}).to_csv(
    RESULTS_DIR / "lj_fit_curve.csv", index=False
)


# =====================================================================
# PART B — BUILD THE Al2 DIMER
# =====================================================================
# TODO:
# Create the coordinates for TWO Al atoms in a 20 A cubic cell.
# Requirements:
#   * initial Al-Al separation = 3.50 A
#   * place the dimer near the center of the box
#   * create an ASE Atoms object with formula "Al2"
#   * use pbc=False
# The variable MUST be named atoms.
#

positions = [
    (8.25, 10, 10) , #3.5 apart, not from center.  10 +/- 1.75
    (11.75, 10, 10) ,
]
atoms = Atoms(
    "Al2",
    positions=positions,
    cell=[CELL_LENGTH_A, CELL_LENGTH_A, CELL_LENGTH_A], #constant at the top
    pbc=False,
)

#never runs now :). well, i guess it shouldn't
# if atoms is None:
#     raise NotImplementedError(
#         "TODO: construct the Al2 initial positions and ASE Atoms object."
#     )


# =====================================================================
# PART C — ATTACH YOUR FITTED POTENTIAL
# =====================================================================
atoms.calc = LennardJones(epsilon=epsilon_fit, sigma=sigma_fit)
initial_distance = atoms.get_distance(0, 1)
initial_energy = atoms.get_potential_energy()
print("\nINITIAL ASE STRUCTURE")
print("=====================")
print(f"Initial Al-Al distance = {initial_distance:.6f} A")
print(f"Initial LJ energy       = {initial_energy:.6f} eV")


# =====================================================================
# PART D — GEOMETRY OPTIMIZATION
# =====================================================================
# TODO:
# Use an ASE geometry optimizer to relax the Al2 dimer.
# Requirements:
#   * choose an ASE geometry-optimization algorithm,
#   * save a trajectory to results/Al2_LJ_optimization.traj,
#   * save a log to results/Al2_LJ_optimization.log,
#   * optimize until fmax <= FMAX_TARGET_EV_PER_A.
# Determine how to import, initialize, and run the optimizer using ASE docs.
#

#https://docs.ase-lib.org/ase/optimize.html checked claude with this page. 
from ase.optimize import BFGS

opt = BFGS(
    atoms,
    trajectory=str(RESULTS_DIR / "Al2_LJ_optimization.traj"),
    logfile=str(RESULTS_DIR / "Al2_LJ_optimization.log"),
)
opt.run(fmax=FMAX_TARGET_EV_PER_A, steps=500)

# Confirm convergence explicitly: largest per-atom force must be <= fmax
#FMAX_TARGET_EV_PER_A = 0.01 (line 45), preset
fmax_final = np.max(np.linalg.norm(atoms.get_forces(), axis=1))
optimization_completed = fmax_final <= FMAX_TARGET_EV_PER_A

if not optimization_completed:
    raise NotImplementedError(
        "TODO: implement and run the ASE geometry optimization, then set "
        "optimization_completed = True."
    )


#ase gui results/Al2_LJ_optimization.traj
#neat

# =====================================================================
# PART E — SAVE THE OPTIMIZED LJ RESULT
# =====================================================================
final_distance = atoms.get_distance(0, 1)
final_energy = atoms.get_potential_energy()
forces = atoms.get_forces()
max_force = float(np.max(np.linalg.norm(forces, axis=1)))

print("\nOPTIMIZED ASE + LJ RESULT")
print("=========================")
print(f"Final Al-Al distance = {final_distance:.6f} A")
print(f"Final LJ energy       = {final_energy:.6f} eV")
print(f"Maximum force         = {max_force:.6f} eV/A")

write(RESULTS_DIR / "Al2_LJ_optimized.xyz", atoms)
lj_results = {
    "initial_distance_A": float(initial_distance),
    "final_distance_A": float(final_distance),
    "initial_energy_eV": float(initial_energy),
    "final_energy_eV": float(final_energy),
    "max_force_eV_per_A": float(max_force),
    "fmax_target_eV_per_A": float(FMAX_TARGET_EV_PER_A),
}
with open(RESULTS_DIR / "lj_optimization_results.json", "w") as f:
    json.dump(lj_results, f, indent=2)

print("\nWrote:")
print("  results/lj_fit_parameters.json")
print("  results/lj_optimization_results.json")
print("  results/Al2_LJ_optimized.xyz")
print("  results/lj_fit.png")
