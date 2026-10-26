#!/usr/bin/env python3
"""
Problem 1 analysis template
Thermal expansion of FCC Al from NPT LAMMPS simulations

This template intentionally DOES:
  * find and read the four LAMMPS production files,
  * label all columns,
  * convert instantaneous volume into an effective FCC lattice parameter,
  * calculate mean / standard deviation / standard error for each temperature,
  * assemble a clean summary table.

You must complete:
  * the lattice-parameter-vs-temperature plot,
  * the linear fit,
  * the coefficient of linear thermal expansion,
  * the physical interpretation.

Expected LAMMPS files in the same directory as this script:
    thermal_100.dat
    thermal_300.dat
    thermal_500.dat
    thermal_700.dat
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent

TEMPERATURES = [100, 300, 500, 700]
N_ATOMS = 256
ATOMS_PER_FCC_CELL = 4


def read_lammps_average_file(path: Path) -> pd.DataFrame:
    """Read the production file written by fix ave/time."""
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    expected_columns = 7
    if data.shape[1] != expected_columns:
        raise ValueError(
            f"{path} has {data.shape[1]} columns; expected {expected_columns}."
        )
    return pd.DataFrame(
        data,
        columns=[
            "step",
            "temperature_K",
            "pressure_bar",
            "volume_A3",
            "Lx_A",
            "Ly_A",
            "Lz_A",
        ],
    )


all_runs = {}
summary_rows = []

for target_T in TEMPERATURES:
    path = ROOT / f"thermal_{target_T}.dat"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.name}. Run all four LAMMPS temperatures first."
        )

    df = read_lammps_average_file(path)

    # FCC: total volume = (N_atoms / 4) * a^3
    df["lattice_parameter_A"] = (
        ATOMS_PER_FCC_CELL * df["volume_A3"] / N_ATOMS
    ) ** (1.0 / 3.0)

    all_runs[target_T] = df
    a_values = df["lattice_parameter_A"].to_numpy()

    summary_rows.append(
        {
            "target_temperature_K": target_T,
            "mean_temperature_K": df["temperature_K"].mean(),
            "mean_pressure_bar": df["pressure_bar"].mean(),
            "mean_volume_A3": df["volume_A3"].mean(),
            "mean_lattice_parameter_A": a_values.mean(),
            "std_lattice_parameter_A": a_values.std(ddof=1),
            "sem_lattice_parameter_A": a_values.std(ddof=1) / np.sqrt(len(a_values)),
        }
    )

summary = pd.DataFrame(summary_rows)
summary.to_csv(ROOT / "thermal_expansion_summary.csv", index=False)

print("\nAveraged production data")
print("========================")
print(summary.to_string(index=False))


# =====================================================================
# TODO 1 — PLOT THE THERMAL EXPANSION
# =====================================================================
# Make a plot of:
#     x = target_temperature_K
#     y = mean_lattice_parameter_A
# Include error bars using sem_lattice_parameter_A.
# Label both axes with units and save:
#     al_thermal_expansion.png
#
# YOUR CODE HERE


# =====================================================================
# TODO 2 — CALCULATE THE LINEAR THERMAL EXPANSION COEFFICIENT
# =====================================================================
# Report alpha in K^-1.
# You may use numpy.polyfit, but Yyou must implement the fit and alpha calculation.
# Add the fitted line to your plot.
#
# YOUR CODE HERE


# =====================================================================
# TODO 3 — REPORT YOUR RESULT
# =====================================================================
# Print:
#   fitted slope da/dT       [A/K]
#   reference lattice a_ref [A]
#   alpha                    [1/K]
#
# YOUR CODE HERE


# =====================================================================
# QUESTIONS TO ADDRESS IN YOUR INTERPRETATION
# =====================================================================
# 1. Does the average lattice parameter increase monotonically with T?
# 2. Why does NPT allow thermal expansion to occur naturally?
# 3. What would change physically if the same simulations were run in NVT?
# 4. Why use time-averaged production data rather than one instantaneous volume?
# 5. What assumptions are made by fitting one linear alpha over 100-700 K?
