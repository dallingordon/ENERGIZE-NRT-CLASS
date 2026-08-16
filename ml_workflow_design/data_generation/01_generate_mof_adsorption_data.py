"""
STAGE 1: DATA GENERATION
========================

Create a synthetic CO2 adsorption dataset for hypothetical MOFs.

The goal is not to reproduce a real adsorption database. The equations below
are teaching equations chosen to create physically sensible trends:

- larger pore volume and surface area increase capacity
- stronger adsorption energy increases affinity
- polar functional groups increase CO2 affinity
- lower temperature increases adsorption
- pressure produces a nonlinear, Langmuir-like isotherm
- defects can change both capacity and affinity

This script writes the RAW dataset used by Stage 2.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. PATHS
# ---------------------------------------------------------------------------

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
PROJECT_DIRECTORY = SCRIPT_DIRECTORY.parent
RAW_DIRECTORY = PROJECT_DIRECTORY / "workflow_data" / "raw"
RAW_DIRECTORY.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 2. DATA-GENERATION CHOICES
# ---------------------------------------------------------------------------

NUMBER_OF_SAMPLES = 2500
RANDOM_SEED = 42

# "log" gives useful coverage over several orders of magnitude.
# Try "linear" and compare the downstream model.
PRESSURE_SAMPLING = "log"

ADD_NOISE = True
NOISE_FRACTION = 0.04

# Optional problems for the processing stage to solve.
ADD_OUTLIERS = False
OUTLIER_FRACTION = 0.01

ADD_MISSING_VALUES = False
MISSING_FRACTION = 0.01


# ---------------------------------------------------------------------------
# 3. GENERATE STRUCTURAL / CHEMICAL DESCRIPTORS
# ---------------------------------------------------------------------------

rng = np.random.default_rng(RANDOM_SEED)

# A hidden porosity variable creates realistic correlations between descriptors.
porosity = rng.beta(2.2, 2.0, NUMBER_OF_SAMPLES)

surface_area = 550 + 3900 * porosity + rng.normal(0, 180, NUMBER_OF_SAMPLES)
surface_area = np.clip(surface_area, 350, 5000)

pore_volume = 0.20 + 1.75 * porosity + rng.normal(0, 0.08, NUMBER_OF_SAMPLES)
pore_volume = np.clip(pore_volume, 0.15, 2.20)

pore_diameter = (
    5.0
    + 14.0 * porosity
    + 8.0 * rng.random(NUMBER_OF_SAMPLES)
    + rng.normal(0, 1.0, NUMBER_OF_SAMPLES)
)
pore_diameter = np.clip(pore_diameter, 5.0, 30.0)

# Highly porous frameworks are generally less dense.
framework_density = 1.75 - 1.05 * porosity + rng.normal(0, 0.08, NUMBER_OF_SAMPLES)
framework_density = np.clip(framework_density, 0.35, 1.80)

# Simple dimensionless descriptor: 0 = weakly polar, 1 = strongly polar.
polarity = rng.uniform(0.0, 1.0, NUMBER_OF_SAMPLES)

# Defect fractions remain relatively small.
defect_fraction = rng.beta(1.5, 7.0, NUMBER_OF_SAMPLES) * 0.20

# Stronger binding is favored by polarity, defects, and somewhat smaller pores.
adsorption_energy = (
    17.0
    + 16.0 * polarity
    + 45.0 * defect_fraction
    + 30.0 / pore_diameter
    + rng.normal(0, 2.0, NUMBER_OF_SAMPLES)
)
adsorption_energy = np.clip(adsorption_energy, 15.0, 48.0)

temperature = rng.uniform(278.0, 348.0, NUMBER_OF_SAMPLES)


# ---------------------------------------------------------------------------
# 4. GENERATE PRESSURE
# ---------------------------------------------------------------------------

MIN_PRESSURE = 0.001
MAX_PRESSURE = 50.0

if PRESSURE_SAMPLING == "log":
    log_pressure = rng.uniform(
        np.log10(MIN_PRESSURE),
        np.log10(MAX_PRESSURE),
        NUMBER_OF_SAMPLES,
    )
    pressure = 10.0 ** log_pressure

elif PRESSURE_SAMPLING == "linear":
    pressure = rng.uniform(MIN_PRESSURE, MAX_PRESSURE, NUMBER_OF_SAMPLES)

else:
    raise ValueError('PRESSURE_SAMPLING must be "log" or "linear".')


# ---------------------------------------------------------------------------
# 5. CREATE SYNTHETIC CO2 UPTAKE
# ---------------------------------------------------------------------------

# Approximate saturation capacity.
pore_efficiency = 0.70 + 0.30 * np.exp(-((pore_diameter - 14.0) / 9.0) ** 2)

maximum_capacity = 0.8 + 3.2 * pore_volume + 0.00065 * surface_area
maximum_capacity *= pore_efficiency * (1.0 - 0.65 * defect_fraction)

# Approximate adsorption affinity.
affinity = (
    0.10
    * np.exp((adsorption_energy - 25.0) / 7.0)
    * np.exp((298.0 - temperature) / 35.0)
    * (0.75 + 0.70 * polarity)
)

# Langmuir-like nonlinear pressure dependence.
true_uptake = maximum_capacity * (affinity * pressure) / (1.0 + affinity * pressure)

# Small low-pressure contribution from defects.
true_uptake += 1.2 * defect_fraction * np.exp(-pressure / 2.0)

measured_uptake = true_uptake.copy()

if ADD_NOISE:
    sigma = 0.03 + NOISE_FRACTION * np.maximum(true_uptake, 0.25)
    measured_uptake += rng.normal(0.0, sigma)

measured_uptake = np.clip(measured_uptake, 0.0, None)


# ---------------------------------------------------------------------------
# 6. OPTIONAL OUTLIERS
# ---------------------------------------------------------------------------

outlier_flag = np.zeros(NUMBER_OF_SAMPLES, dtype=bool)

if ADD_OUTLIERS:
    number_of_outliers = max(1, int(OUTLIER_FRACTION * NUMBER_OF_SAMPLES))
    indices = rng.choice(NUMBER_OF_SAMPLES, number_of_outliers, replace=False)
    measured_uptake[indices] *= rng.uniform(1.7, 2.5, number_of_outliers)
    outlier_flag[indices] = True


# ---------------------------------------------------------------------------
# 7. BUILD THE RAW DATAFRAME
# ---------------------------------------------------------------------------

raw_df = pd.DataFrame(
    {
        "Sample_ID": [f"MOF_SAMPLE_{i:05d}" for i in range(1, NUMBER_OF_SAMPLES + 1)],
        "Surface_Area_m2_g": surface_area,
        "Pore_Volume_cm3_g": pore_volume,
        "Average_Pore_Diameter_A": pore_diameter,
        "Framework_Density_g_cm3": framework_density,
        "Adsorption_Energy_kJ_mol": adsorption_energy,
        "Functional_Group_Polarity": polarity,
        "Defect_Fraction": defect_fraction,
        "Temperature_K": temperature,
        "Pressure_bar": pressure,
        "CO2_Uptake_mmol_g": measured_uptake,
        "Synthetic_Outlier_Flag": outlier_flag,
    }
)

FEATURE_COLUMNS = [
    "Surface_Area_m2_g",
    "Pore_Volume_cm3_g",
    "Average_Pore_Diameter_A",
    "Framework_Density_g_cm3",
    "Adsorption_Energy_kJ_mol",
    "Functional_Group_Polarity",
    "Defect_Fraction",
    "Temperature_K",
    "Pressure_bar",
]


# ---------------------------------------------------------------------------
# 8. OPTIONAL MISSING VALUES
# ---------------------------------------------------------------------------

if ADD_MISSING_VALUES:
    number_of_missing_cells = int(
        MISSING_FRACTION * NUMBER_OF_SAMPLES * len(FEATURE_COLUMNS)
    )

    for _ in range(number_of_missing_cells):
        row = rng.integers(0, NUMBER_OF_SAMPLES)
        column = rng.choice(FEATURE_COLUMNS)
        raw_df.loc[row, column] = np.nan


# ---------------------------------------------------------------------------
# 9. SAVE RAW DATA AND SETTINGS
# ---------------------------------------------------------------------------

raw_file = RAW_DIRECTORY / "synthetic_mof_adsorption_raw.csv"
raw_df.to_csv(raw_file, index=False)

settings = {
    "NUMBER_OF_SAMPLES": NUMBER_OF_SAMPLES,
    "PRESSURE_SAMPLING": PRESSURE_SAMPLING,
    "ADD_NOISE": ADD_NOISE,
    "NOISE_FRACTION": NOISE_FRACTION,
    "ADD_OUTLIERS": ADD_OUTLIERS,
    "ADD_MISSING_VALUES": ADD_MISSING_VALUES,
}

with open(RAW_DIRECTORY / "data_generation_settings.json", "w") as file:
    json.dump(settings, file, indent=4)


# ---------------------------------------------------------------------------
# 10. DIAGNOSTIC FIGURE
# ---------------------------------------------------------------------------

figure, axes = plt.subplots(2, 2, figsize=(12, 9))

axes[0, 0].hist(raw_df["Pressure_bar"], bins=40)
axes[0, 0].set_xlabel("Pressure (bar)")
axes[0, 0].set_ylabel("Count")
axes[0, 0].set_title("Pressure on Linear Axis")

axes[0, 1].hist(np.log10(raw_df["Pressure_bar"]), bins=40)
axes[0, 1].set_xlabel("log10(Pressure / bar)")
axes[0, 1].set_ylabel("Count")
axes[0, 1].set_title("Pressure on Log Axis")

axes[1, 0].scatter(raw_df["Pressure_bar"], raw_df["CO2_Uptake_mmol_g"], s=10)
axes[1, 0].set_xscale("log")
axes[1, 0].set_xlabel("Pressure (bar)")
axes[1, 0].set_ylabel("CO2 Uptake (mmol/g)")
axes[1, 0].set_title("Synthetic Adsorption Isotherm Data")

axes[1, 1].scatter(raw_df["Pore_Volume_cm3_g"], raw_df["CO2_Uptake_mmol_g"], s=10)
axes[1, 1].set_xlabel("Pore Volume (cm3/g)")
axes[1, 1].set_ylabel("CO2 Uptake (mmol/g)")
axes[1, 1].set_title("Pore Volume vs Uptake")

figure.suptitle("Stage 1: Raw Synthetic MOF Adsorption Data", fontsize=15)
figure.tight_layout()
figure.savefig(RAW_DIRECTORY / "01_raw_data_diagnostics.png", dpi=200)
plt.show()


# ---------------------------------------------------------------------------
# 11. SUMMARY
# ---------------------------------------------------------------------------

print("=" * 75)
print("STAGE 1 COMPLETE: DATA GENERATION")
print("=" * 75)
print("Raw dataset:", raw_file)
print("Shape:", raw_df.shape)
print()
print(raw_df.head())
print()
print("Next: run 02_data_processing/02_process_mof_adsorption_data.py")
