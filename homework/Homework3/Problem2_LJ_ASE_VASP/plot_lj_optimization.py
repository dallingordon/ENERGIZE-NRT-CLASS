#!/usr/bin/env python3
"""
Plot the ASE LJ energy curve with each BFGS optimization step overlaid,
and mark the analytical LJ minimum r_eq = 2^(1/6) * sigma.

Run after fit_lj_and_optimize_ase_template.py:
    python plot_lj_optimization.py
Writes results/lj_optimization_path.png
"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from ase.io import read
from ase.calculators.lj import LennardJones

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"

params = json.load(open(RESULTS_DIR / "lj_fit_parameters.json"))
eps, sig = params["epsilon_eV"], params["sigma_A"]
r_eq = params["analytic_equilibrium_distance_A"]

# Energy curve from the SAME ASE calculator the optimizer used
r_grid = np.linspace(2.3, 4.0, 400)
e_grid = []
for r in r_grid:
    dimer = Atoms("Al2", positions=[(0, 0, 0), (r, 0, 0)], pbc=False)
    dimer.calc = LennardJones(epsilon=eps, sigma=sig)
    e_grid.append(dimer.get_potential_energy())

# Each BFGS step from the saved trajectory
frames = read(RESULTS_DIR / "Al2_LJ_optimization.traj", index=":")
r_steps = [f.get_distance(0, 1) for f in frames]
e_steps = [f.get_potential_energy() for f in frames]

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(r_grid, e_grid, color="0.3", label="ASE LennardJones (fitted ε, σ)")
ax.plot(r_steps, e_steps, "o--", color="tab:orange", label="BFGS steps")
for i, (r, e) in enumerate(zip(r_steps, e_steps)):
    ax.annotate(str(i), (r, e), textcoords="offset points", xytext=(6, 6), fontsize=8)
ax.axvline(r_eq, linestyle=":", color="tab:blue",
           label=f"Analytical minimum 2^(1/6)σ = {r_eq:.4f} Å")
ax.plot(r_steps[-1], e_steps[-1], "*", ms=14, color="tab:red",
        label=f"ASE final = {r_steps[-1]:.4f} Å")
ax.set_xlabel("Al–Al distance (Å)")
ax.set_ylabel("LJ energy (eV)")
ax.set_title("Al$_2$ geometry optimization on the fitted LJ potential")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "lj_optimization_path.png", dpi=200)
print("Wrote results/lj_optimization_path.png")
