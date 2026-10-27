#!/usr/bin/env python3
"""
Analyze five independent NPT simulations of FCC Al.

Place this script in the same local directory as:
    thermal_<temperature>.dat

The script automatically:
  * finds all five files,
  * reads block-averaged T, P, and V,
  * calculates average volume and fluctuations,
  * fits V(T),
  * calculates volumetric beta,
  * estimates linear alpha = beta/3 for cubic Al,
  * saves a plot and summary CSV.
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
REFERENCE_TEMPERATURE_K = 300.0


def temperature_from_filename(path):
    match = re.fullmatch(r"thermal_([+-]?\d+(?:\.\d+)?)\.dat", path.name)
    if not match:
        raise ValueError(f"Could not extract temperature from {path.name}")
    return float(match.group(1))


def read_lammps_average_file(path):
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != 4:
        raise ValueError(
            f"{path.name} has {data.shape[1]} numeric columns; expected 4."
        )
    return pd.DataFrame(
        data,
        columns=["step", "temperature_K", "pressure_bar", "volume_A3"],
    )


files = sorted(ROOT.glob("thermal_*.dat"), key=temperature_from_filename)

if len(files) != 5:
    raise RuntimeError(
        f"Expected exactly 5 thermal_*.dat files, but found {len(files)}.\n"
        f"Files found: {[p.name for p in files]}"
    )

rows = []
for path in files:
    target_T = temperature_from_filename(path)
    df = read_lammps_average_file(path)
    V = df["volume_A3"].to_numpy()

    rows.append(
        {
            "target_temperature_K": target_T,
            "mean_temperature_K": df["temperature_K"].mean(),
            "mean_pressure_bar": df["pressure_bar"].mean(),
            "mean_volume_A3": V.mean(),
            "std_volume_A3": V.std(ddof=1),
            "sem_volume_A3": V.std(ddof=1) / np.sqrt(len(V)),
            "n_block_averages": len(V),
        }
    )

summary = (
    pd.DataFrame(rows)
    .sort_values("target_temperature_K")
    .reset_index(drop=True)
)

T = summary["target_temperature_K"].to_numpy()
V = summary["mean_volume_A3"].to_numpy()

# Linear fit: V(T) = intercept + slope*T
slope, intercept = np.polyfit(T, V, 1)
V_fit = intercept + slope * T

# Use the fitted volume at 300 K as the reference volume.
V_ref = intercept + slope * REFERENCE_TEMPERATURE_K

# Volumetric coefficient beta = (1/V) dV/dT
beta = slope / V_ref

# For an isotropic cubic material, beta ~ 3 alpha.
alpha = beta / 3.0

summary["fit_volume_A3"] = V_fit
summary.to_csv(ROOT / "thermal_expansion_summary.csv", index=False)

# Plot mean volume with +/- 1 SD of the production block averages.
plt.figure(figsize=(7.2, 5.2))
plt.errorbar(
    summary["target_temperature_K"],
    summary["mean_volume_A3"],
    yerr=summary["std_volume_A3"],
    fmt="o",
    capsize=4,
    label="NPT mean volume ± 1 SD",
)

T_line = np.linspace(T.min(), T.max(), 300)
V_line = intercept + slope * T_line
plt.plot(T_line, V_line, label="Linear fit")

plt.xlabel("Target temperature (K)")
plt.ylabel(r"Average simulation-cell volume ($\AA^3$)")
plt.title("Thermal Expansion of FCC Al from NPT MD")
plt.legend()

annotation = (
    f"dV/dT = {slope:.4f} $\\AA^3$/K\n"
    f"$\\beta$ = {beta:.3e} K$^{{-1}}$\n"
    f"$\\alpha \\approx \\beta/3$ = {alpha:.3e} K$^{{-1}}$"
)
plt.text(
    0.04, 0.96, annotation,
    transform=plt.gca().transAxes,
    va="top"
)

plt.tight_layout()
plt.savefig(ROOT / "al_thermal_expansion.png", dpi=200)
plt.close()

print("\nTHERMAL EXPANSION RESULTS")
print("=========================\n")
print(
    summary[
        [
            "target_temperature_K",
            "mean_temperature_K",
            "mean_pressure_bar",
            "mean_volume_A3",
            "std_volume_A3",
        ]
    ].to_string(index=False)
)

print("\nLinear fit")
print("----------")
print(f"dV/dT                = {slope:.6f} A^3/K")
print(f"V(300 K) from fit    = {V_ref:.6f} A^3")
print(f"Volumetric beta      = {beta:.6e} 1/K")
print(f"Approx. linear alpha = {alpha:.6e} 1/K")

print("\nCreated:")
print("  al_thermal_expansion.png")
print("  thermal_expansion_summary.csv")
