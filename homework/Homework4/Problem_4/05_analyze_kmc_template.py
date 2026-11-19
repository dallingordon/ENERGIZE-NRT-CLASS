#!/usr/bin/env python3
"""
Question 4E-F: Calculate vacancy MSD, diffusivity, and Arrhenius activation energy.

STUDENT TASKS
-------------
1. Calculate the ensemble-averaged vacancy MSD for every temperature.
2. Use MSD = 6 D t in 3D to determine D_v at each temperature.
3. Plot the MSD curves and their fitted long-time regions.
4. Fit ln(D_v) versus 1/T and extract Ea_KMC.
5. Compare Ea_KMC, Ea_NEB, and the experimental vacancy migration Ea.
6. Compare KMC vacancy diffusivities with the experimental Arrhenius relation.

Experimental benchmark used here:
  Khellaf, Seeger, and Emrick, Materials Transactions 43 (2002) 186-198.
  Monovacancy diffusivity:
      D_exp = 6e-6 * exp[-0.61 eV / (k_B T)]  m^2/s
  Reported migration enthalpy: 0.61 +/- 0.02 eV.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
KMC_DIR = ROOT / "kmc"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

KB_EV_PER_K = 8.617333262145e-5
EXP_EA_EV = 0.61
EXP_EA_UNCERTAINTY_EV = 0.02
EXP_D0_M2_S = 6.0e-6
FIT_START_FRACTION = 0.30

neb_file = RESULTS_DIR / "neb_results.json"
if not neb_file.exists():
    raise FileNotFoundError("Missing results/neb_results.json")
with open(neb_file) as f:
    neb_results = json.load(f)
EA_NEB_EV = float(neb_results["migration_barrier_eV"])

kmc_files = sorted(KMC_DIR.glob("kmc_T*K.npz"))
if len(kmc_files) != 5:
    raise RuntimeError(
        f"Expected five KMC files in kmc/, but found {len(kmc_files)}."
    )

summary_rows = []
msd_data = []

for path in kmc_files:
    data = np.load(path)
    temperature_K = float(data["temperature_K"])
    times_s = np.asarray(data["times_s"], dtype=float)
    positions_m = np.asarray(data["positions_m"], dtype=float)

    # Shapes:
    #   times_s     -> (n_replicas, n_samples)
    #   positions_m -> (n_replicas, n_samples, 3)

    # ========================================================================
    # TODO 1 — CALCULATE ENSEMBLE-AVERAGED MSD(t)
    # ========================================================================
    # For each replica, calculate displacement relative to that replica's
    # initial vacancy position. Then calculate squared displacement:
    #
    #     r^2(t) = dx^2 + dy^2 + dz^2
    #
    # Finally average r^2(t) over all replicas to obtain MSD(t).
    # Also calculate the mean physical KMC time at each sampled event index.
    #
    # Required variables:
    #     mean_time_s   -> shape (n_samples,)
    #     msd_m2        -> shape (n_samples,)
    # ========================================================================

    displacement_m = positions_m - positions_m[:, 0:1, :]      # (128, n, 3)
    r2_m2 = np.sum(displacement_m**2, axis=-1)                 # (128, n)

    mean_time_s = times_s.mean(axis=0) 
    msd_m2 = r2_m2.mean(axis=0)

    if mean_time_s is None or msd_m2 is None:
        raise NotImplementedError(
            "TODO 1: calculate mean_time_s and ensemble-averaged msd_m2."
        )

    mean_time_s = np.asarray(mean_time_s, dtype=float)
    msd_m2 = np.asarray(msd_m2, dtype=float)

    if mean_time_s.shape != msd_m2.shape:
        raise RuntimeError("mean_time_s and msd_m2 must have the same shape.")

    # Use the later 70% of the trajectory for a long-time diffusion fit.
    fit_start = max(1, int(FIT_START_FRACTION * len(mean_time_s)))
    fit_t = mean_time_s[fit_start:]
    fit_msd = msd_m2[fit_start:]

    # ========================================================================
    # TODO 2 — EXTRACT THE VACANCY DIFFUSIVITY
    # ========================================================================
    # In three dimensions:
    #
    #     MSD(t) = 6 D_v t
    #
    # Fit a line to fit_msd versus fit_t. Store:
    #     msd_slope_m2_s
    #     msd_intercept_m2
    #     diffusivity_m2_s
    #
    # Use the slope to calculate D_v.
    # ========================================================================
    fit_out = np.polyfit(fit_t, fit_msd, 1)
    msd_slope_m2_s = fit_out[0]
    msd_intercept_m2 = fit_out[1]
    diffusivity_m2_s = msd_slope_m2_s / 6.0



    if (msd_slope_m2_s is None or msd_intercept_m2 is None
            or diffusivity_m2_s is None):
        raise NotImplementedError(
            "TODO 2: fit MSD(t) and calculate the vacancy diffusivity."
        )

    msd_slope_m2_s = float(msd_slope_m2_s)
    msd_intercept_m2 = float(msd_intercept_m2)
    diffusivity_m2_s = float(diffusivity_m2_s)

    if diffusivity_m2_s <= 0:
        raise RuntimeError("Calculated diffusivity must be positive.")

    exp_diffusivity_m2_s = EXP_D0_M2_S * np.exp(
        -EXP_EA_EV / (KB_EV_PER_K * temperature_K)
    )

    summary_rows.append({
        "temperature_K": temperature_K,
        "D_KMC_m2_s": diffusivity_m2_s,
        "D_experiment_m2_s": exp_diffusivity_m2_s,
        "KMC_to_experiment_ratio": diffusivity_m2_s / exp_diffusivity_m2_s,
        "MSD_fit_slope_m2_s": msd_slope_m2_s,
        "fit_start_time_s": float(fit_t[0]),
        "fit_end_time_s": float(fit_t[-1]),
    })

    msd_data.append({
        "temperature_K": temperature_K,
        "time_s": mean_time_s,
        "msd_m2": msd_m2,
        "fit_t_s": fit_t,
        "fit_msd_m2": msd_slope_m2_s * fit_t + msd_intercept_m2,
    })

summary = pd.DataFrame(summary_rows).sort_values("temperature_K").reset_index(drop=True)


# ============================================================================
# TODO 3 — PLOT MSD(t) FOR ALL FIVE TEMPERATURES
# ============================================================================
# Make one figure containing the ensemble-averaged MSD curves for all five
# temperatures. Show the fitted long-time line for each temperature as well.
# Label axes with units and save:
#
#     results/vacancy_msd.png
# ============================================================================

fig, ax = plt.subplots(figsize=(7, 5))
for i, d in enumerate(sorted(msd_data, key=lambda x: x["temperature_K"])):
    c = plt.cm.viridis(i / 4)
    ax.plot(d["time_s"], d["msd_m2"], color=c, lw=1.5, label=f'{d["temperature_K"]:.0f} K')
    ax.plot(d["fit_t_s"], d["fit_msd_m2"], "--", color="k", lw=1)
ax.plot([], [], "--", color="k", label="Linear fit (MSD = 6Dt + b)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("KMC time (s)")
ax.set_ylabel("Vacancy MSD (m$^2$)")
ax.set_title("Vacancy MSD in FCC Al (on-lattice KMC)")
ax.legend()
fig.tight_layout()
fig.savefig(RESULTS_DIR / "vacancy_msd.png", dpi=200)
plt.close(fig)


if not (RESULTS_DIR / "vacancy_msd.png").exists():
    raise FileNotFoundError(
        "TODO 3: save the MSD figure as results/vacancy_msd.png"
    )


# ============================================================================
# TODO 4 — ARRHENIUS FIT AND KMC ACTIVATION ENERGY
# ============================================================================
# The Arrhenius relation is:
#
#     D = D0 exp[-Ea/(k_B T)]
#
# Therefore:
#
#     ln(D) = ln(D0) - (Ea/k_B) * (1/T)
#
# Fit ln(D_KMC) versus 1/T and determine:
#     arrhenius_slope_K
#     arrhenius_intercept
#     EA_KMC_EV
#
# Use k_B in eV/K so Ea is obtained directly in eV.
# ============================================================================
temperatures_K = summary["temperature_K"].to_numpy(dtype=float)
d_kmc = summary["D_KMC_m2_s"].to_numpy(dtype=float)
d_exp = summary["D_experiment_m2_s"].to_numpy(dtype=float)

inverse_temperature_Kinv = 1.0 / temperatures_K
ln_d_kmc = np.log(d_kmc)

arrhenius_slope_K, arrhenius_intercept = np.polyfit(inverse_temperature_Kinv, ln_d_kmc, 1)
EA_KMC_EV = -arrhenius_slope_K * KB_EV_PER_K
#arrhenius_slope_K = None
#arrhenius_intercept = None
#EA_KMC_EV = None

if arrhenius_slope_K is None or arrhenius_intercept is None or EA_KMC_EV is None:
    raise NotImplementedError(
        "TODO 4: fit ln(D) versus 1/T and calculate the KMC activation energy."
    )

arrhenius_slope_K = float(arrhenius_slope_K)
arrhenius_intercept = float(arrhenius_intercept)
EA_KMC_EV = float(EA_KMC_EV)


# ============================================================================
# TODO 5 — ARRHENIUS / EXPERIMENT COMPARISON PLOT
# ============================================================================
# Plot ln(D) versus 1/T for:
#   * your KMC diffusivities
#   * your fitted KMC Arrhenius line
#   * the experimental vacancy diffusivities at the same temperatures
#
# Save:
#     results/vacancy_arrhenius.png
# ============================================================================

invT_line = np.linspace(inverse_temperature_Kinv.min(), inverse_temperature_Kinv.max(), 200)
ln_d_exp_line = np.log(EXP_D0_M2_S) - EXP_EA_EV / KB_EV_PER_K * invT_line

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(inverse_temperature_Kinv, ln_d_kmc, "o", color="tab:blue", ms=8, label="KMC")
ax.plot(invT_line, arrhenius_intercept + arrhenius_slope_K * invT_line, "-", color="tab:blue",
        label=f"KMC fit, $E_a$ = {EA_KMC_EV:.3f} eV")
ax.plot(inverse_temperature_Kinv, np.log(d_exp), "s", color="tab:red", ms=8, label="Experiment (same T)")
ax.plot(invT_line, ln_d_exp_line, "--", color="tab:red",
        label=f"Experiment, $E_a$ = {EXP_EA_EV:.2f} eV")
ax.set_xlabel("1/T (K$^{-1}$)")
ax.set_ylabel("ln($D_v$ / m$^2$ s$^{-1}$)")
ax.set_title("Vacancy diffusivity: KMC vs experiment")
ax.legend()
fig.tight_layout()
fig.savefig(RESULTS_DIR / "vacancy_arrhenius.png", dpi=200)
plt.close(fig)


if not (RESULTS_DIR / "vacancy_arrhenius.png").exists():
    raise FileNotFoundError(
        "TODO 5: save the Arrhenius plot as results/vacancy_arrhenius.png"
    )

summary.to_csv(RESULTS_DIR / "kmc_diffusivity_summary.csv", index=False)

comparison = {
    "Ea_NEB_eV": EA_NEB_EV,
    "Ea_KMC_eV": EA_KMC_EV,
    "Ea_experiment_eV": EXP_EA_EV,
    "Ea_experiment_uncertainty_eV": EXP_EA_UNCERTAINTY_EV,
    "experimental_D0_m2_s": EXP_D0_M2_S,
    "NEB_minus_experiment_eV": EA_NEB_EV - EXP_EA_EV,
    "KMC_minus_NEB_eV": EA_KMC_EV - EA_NEB_EV,
    "KMC_minus_experiment_eV": EA_KMC_EV - EXP_EA_EV,
}

with open(RESULTS_DIR / "activation_energy_comparison.json", "w") as f:
    json.dump(comparison, f, indent=2)

print("\nVACANCY DIFFUSION SUMMARY")
print("=========================")
print(summary.to_string(index=False))
print("\nActivation energies")
print("-------------------")
print(f"NEB:         {EA_NEB_EV:.6f} eV")
print(f"KMC fit:     {EA_KMC_EV:.6f} eV")
print(f"Experiment:  {EXP_EA_EV:.2f} +/- {EXP_EA_UNCERTAINTY_EV:.2f} eV")
print("\nCreated:")
print("  results/vacancy_msd.png")
print("  results/vacancy_arrhenius.png")
print("  results/kmc_diffusivity_summary.csv")
print("  results/activation_energy_comparison.json")
