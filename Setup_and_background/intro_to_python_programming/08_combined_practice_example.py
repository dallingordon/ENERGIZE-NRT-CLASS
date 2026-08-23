"""
08_combined_practice_example.py

Purpose
-------
This script combines several ideas from the previous examples.

It demonstrates a simple scientific-data workflow:

    1. Store data in NumPy arrays
    2. Perform calculations
    3. Store results in a pandas DataFrame
    4. Filter the data
    5. Save the results to a CSV file
    6. Create a figure

This is similar to the structure of many scripts used in research.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. CREATE SYNTHETIC DATA
# ---------------------------------------------------------------------------

# Imagine that several molecular dynamics simulations were performed at
# different temperatures.

temperature = np.array([300, 500, 700, 900, 1100, 1300])

# These are example energies in eV/atom.
energy = np.array([-3.42, -3.35, -3.20, -2.95, -2.60, -2.20])


# ---------------------------------------------------------------------------
# 2. CALCULATE A NEW QUANTITY
# ---------------------------------------------------------------------------

# Here we calculate the energy relative to the lowest-energy value.

minimum_energy = np.min(energy)

relative_energy = energy - minimum_energy


# ---------------------------------------------------------------------------
# 3. CREATE A PANDAS DATAFRAME
# ---------------------------------------------------------------------------

data = {
    "Temperature_K": temperature,
    "Energy_eV_atom": energy,
    "Relative_Energy_eV_atom": relative_energy
}

df = pd.DataFrame(data)

print("Complete dataset:")
print(df)

print()


# ---------------------------------------------------------------------------
# 4. FILTER THE DATA
# ---------------------------------------------------------------------------

# Select simulations performed at 900 K or above.

high_temperature_data = df[df["Temperature_K"] >= 900]

print("High-temperature data:")
print(high_temperature_data)

print()


# ---------------------------------------------------------------------------
# 5. SAVE THE DATA
# ---------------------------------------------------------------------------

df.to_csv("combined_example_results.csv", index=False)

print("Saved combined_example_results.csv")

print()


# ---------------------------------------------------------------------------
# 6. CREATE A FIGURE
# ---------------------------------------------------------------------------

plt.figure()

plt.plot(
    df["Temperature_K"],
    df["Relative_Energy_eV_atom"],
    marker="o"
)

plt.xlabel("Temperature (K)")
plt.ylabel("Relative Energy (eV/atom)")
plt.title("Relative Energy versus Temperature")

plt.tight_layout()

plt.savefig("combined_example_figure.png", dpi=200)

plt.show()


# ---------------------------------------------------------------------------
# PRACTICE EXERCISE
# ---------------------------------------------------------------------------

# Modify this script so that it also calculates a column called:
#
# Energy_meV_atom
#
# Remember:
#
# 1 eV = 1000 meV
#
# Then save the updated DataFrame to the CSV file.
#
# One possible solution is shown below.

df["Energy_meV_atom"] = df["Energy_eV_atom"] * 1000

df.to_csv("combined_example_results_with_meV.csv", index=False)

print()
print("Practice exercise:")
print("Added Energy_meV_atom and saved combined_example_results_with_meV.csv")
