#!/usr/bin/env python3
"""
Complete comparison script for Problem 2.

EXPECTED DIRECTORY STRUCTURE
Problem2_LJ_ASE_VASP/
|-- compare_lj_vasp.py
|-- results/
|   |-- lj_fit_parameters.json
|   `-- lj_optimization_results.json
`-- vasp/
    |-- CONTCAR
    `-- OUTCAR

The script compares:
  1. analytical minimum of the fitted LJ potential,
  2. ASE geometry optimization using the fitted LJ potential,
  3. final VASP/PBE geometry optimization.

Absolute LJ and VASP total energies are NOT compared because their energy
zeros are not equivalent.
"""

from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ase.io import read

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
VASP_DIR = ROOT / "vasp"


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Check the directory structure described "
            "at the top of this script."
        )
    with open(path) as f:
        return json.load(f)


def parse_vasp_toten_history(outcar_path):
    energies = []
    pattern = re.compile(
        r"free\s+energy\s+TOTEN\s*=\s*([+-]?\d+(?:\.\d*)?(?:[Ee][+-]?\d+)?)"
    )
    with open(outcar_path, errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                energies.append(float(m.group(1)))
    return np.asarray(energies, dtype=float)


def parse_vasp_max_force_history(outcar_path):
    max_forces = []
    with open(outcar_path, errors="ignore") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        if "TOTAL-FORCE (eV/Angst)" in lines[i]:
            i += 2
            forces = []
            while i < len(lines):
                line = lines[i].strip()
                if not line or line.startswith("-"):
                    break
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        fx, fy, fz = map(float, parts[3:6])
                        forces.append([fx, fy, fz])
                    except ValueError:
                        pass
                i += 1
            if forces:
                norms = np.linalg.norm(np.asarray(forces), axis=1)
                max_forces.append(float(np.max(norms)))
        i += 1
    return np.asarray(max_forces, dtype=float)


fit = load_json(RESULTS_DIR / "lj_fit_parameters.json")
lj = load_json(RESULTS_DIR / "lj_optimization_results.json")
analytic_lj_distance = float(fit["analytic_equilibrium_distance_A"])
ase_lj_distance = float(lj["final_distance_A"])

contcar = VASP_DIR / "CONTCAR"
outcar = VASP_DIR / "OUTCAR"
if not contcar.exists():
    raise FileNotFoundError("Missing vasp/CONTCAR. Complete the VASP optimization first.")
if not outcar.exists():
    raise FileNotFoundError("Missing vasp/OUTCAR. Complete the VASP optimization first.")

vasp_atoms = read(contcar)
if len(vasp_atoms) != 2:
    raise ValueError(f"Expected Al2 CONTCAR with 2 atoms; found {len(vasp_atoms)}.")
vasp_distance = float(vasp_atoms.get_distance(0, 1, mic=True))

difference = ase_lj_distance - vasp_distance
percent_difference = 100.0 * abs(difference) / vasp_distance

comparison = pd.DataFrame({
    "method": ["Analytical LJ minimum", "ASE + fitted LJ", "VASP/PBE"],
    "Al_Al_distance_A": [analytic_lj_distance, ase_lj_distance, vasp_distance],
})
comparison.to_csv(RESULTS_DIR / "lj_vs_vasp_comparison.csv", index=False)

print("\nAl2 EQUILIBRIUM-DISTANCE COMPARISON")
print("===================================")
print(comparison.to_string(index=False))
print("\nASE-LJ vs VASP")
print("--------------")
print(f"Signed difference = {difference:.6f} A")
print(f"Percent difference = {percent_difference:.3f} %")

plt.figure(figsize=(7, 4.5))
plt.bar(comparison["method"], comparison["Al_Al_distance_A"])
plt.ylabel("Optimized Al-Al distance (A)")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "lj_vs_vasp_comparison.png", dpi=200)
plt.close()

energies = parse_vasp_toten_history(outcar)
max_forces = parse_vasp_max_force_history(outcar)

if len(energies) > 0:
    ionic_step = np.arange(1, len(energies)+1)
    relative_energy = energies - energies[-1]
    plt.figure(figsize=(6.5,4.5))
    plt.plot(ionic_step, relative_energy, marker="o")
    plt.xlabel("Ionic step")
    plt.ylabel("Energy relative to final step (eV)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "vasp_energy_convergence.png", dpi=200)
    plt.close()

if len(max_forces) > 0:
    force_step = np.arange(1, len(max_forces)+1)
    plt.figure(figsize=(6.5,4.5))
    plt.plot(force_step, max_forces, marker="o")
    plt.xlabel("Ionic step")
    plt.ylabel("Maximum atomic force (eV/A)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "vasp_force_convergence.png", dpi=200)
    plt.close()

summary = {
    "analytic_lj_distance_A": analytic_lj_distance,
    "ase_lj_distance_A": ase_lj_distance,
    "vasp_pbe_distance_A": vasp_distance,
    "ase_lj_minus_vasp_A": difference,
    "absolute_percent_difference": percent_difference,
    "vasp_ionic_steps_from_toten": int(len(energies)),
    "vasp_force_tables": int(len(max_forces)),
}
with open(RESULTS_DIR / "comparison_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nWrote comparison outputs to results/.")
print("Note: absolute LJ and VASP total energies are intentionally not compared.")
