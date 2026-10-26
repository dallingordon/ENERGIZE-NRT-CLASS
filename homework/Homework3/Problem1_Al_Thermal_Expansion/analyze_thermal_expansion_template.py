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
T = summary["target_temperature_K"].to_numpy()
a = summary["mean_lattice_parameter_A"].to_numpy()
a_err = summary["sem_lattice_parameter_A"].to_numpy()

fig, (ax, ax_res) = plt.subplots(
    2, 1, figsize=(7, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]}
)
ax.errorbar(T, a, yerr=a_err, fmt="o", ms=6, capsize=4, zorder=3, label="MD average (± SEM)")
ax.set_ylabel("Lattice parameter a (Å)")
ax.set_title("FCC Al thermal expansion (EAM, NPT)")
ax.legend()


# =====================================================================
# TODO 2 — CALCULATE THE LINEAR THERMAL EXPANSION COEFFICIENT
# =====================================================================
# Report alpha in K^-1.
# You may use numpy.polyfit, but Yyou must implement the fit and alpha calculation.
# Add the fitted line to your plot.
#
# Linear fit a(T) = a_ref + m*T  (polyfit returns [slope, intercept])
m, a_ref = np.polyfit(T, a, 1) # the fit
alpha = m / a_ref  # alpha calc

T_line = np.linspace(T.min(), T.max(), 100)
ax.plot(T_line, a_ref + m * T_line, "--", label="linear fit")
ax.legend()

# Residual panel: data minus fit, in milli-angstroms, so the SEM bars are visible
residual_mA = (a - (a_ref + m * T)) * 1000
ax_res.axhline(0, color="C1", ls="--", lw=1)
ax_res.errorbar(T, residual_mA, yerr=a_err * 1000, fmt="o", ms=6, capsize=4)
ax_res.set_ylabel("data − fit (mÅ)")
ax_res.set_xlabel("Temperature (K)")
plt.tight_layout()
#plt.savefig(ROOT / "al_thermal_expansion.png", dpi=150)
plt.show()


# =====================================================================
# TODO 3 — REPORT YOUR RESULT
# =====================================================================
# Print:
#   fitted slope da/dT       [A/K]
#   reference lattice a_ref [A]
#   alpha                    [1/K]
#
print("\nLinear thermal expansion fit: a(T) = a_ref + (da/dT) * T")
print("=======================================================")
print(f"fitted slope da/dT      = {m:.4e} A/K")
print(f"reference lattice a_ref = {a_ref:.5f} A")
print(f"alpha                   = {alpha:.4e} 1/K")


##EIn a short response, explain why NPT is the appropriate ensemble for this property. Explicitly
##contrast it with NVT, where the simulation-cell volume is fixed. Your response should connect the
##controlled ensemble variables to the physical quantity being measured.
## Studying thermal expansion means we are interested in how a material expands, how its volume changes, as temperature increases.  
## So we can see provided a set of inputs what our output would be.  We care about volume explicitly, we want to see at a particular temperature.
## So lets let it vary, and pick our temperatures.  NVT would mean we selected temperature and volume which just writing that out seems to go against the question we are asking. 
## in the case of NVT we could still pick our temperature, but we would also have to pick our volume, and pressure would be the property we would meassure. 
## we would basically be constraining the thermal expansion to whatever volume we chose.  It wouldn't give us the output we wanted to anwer our question.
##

# =====================================================================
# QUESTIONS TO ADDRESS IN YOUR INTERPRETATION
# =====================================================================
# 1. Does the average lattice parameter increase monotonically with T?
#  It does, but not linearly.  it is growing faster than a linear fit line.
# 2. Why does NPT allow thermal expansion to occur naturally?
# Since volume isn't held constant and pressure and temperature are that is the variable that has to change to keep those constant. Increase in temperature increases the amplitude at which the atoms vibrate and since it is easier to move ataoms apart than closer together at a certain point, expansion happens. 
# 3. What would change physically if the same simulations were run in NVT?
#  Pressure would increase with temperature since now volume is held constant. 
# 4. Why use time-averaged production data rather than one instantaneous volume?
# There are fluctuations happening all the time, and the parameter we actually care about expressed as an average overcomes those fluctuations.  Single points can be high or low, less likely to capture and express the property we care about.
# 5. What assumptions are made by fitting one linear alpha over 100-700 K?
# That the volume grows linearly with temperature.  It appears not to (as mentioned in 1 it seems to grow faster than linearly).
