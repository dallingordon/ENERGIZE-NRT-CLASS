"""
06_basic_plotting.py

Purpose
-------
This script introduces basic plotting with matplotlib.

Scientific Python scripts often create figures so that trends can be
understood visually.

This script introduces:
    1. Line plots
    2. Scatter plots
    3. Axis labels
    4. Figure titles
    5. Saving figures
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. CREATE SOME SIMPLE DATA
# ---------------------------------------------------------------------------

temperature = np.array([300, 600, 900, 1200, 1500])
energy = np.array([-3.40, -3.25, -3.00, -2.60, -2.10])


# ---------------------------------------------------------------------------
# 2. MAKE A LINE PLOT
# ---------------------------------------------------------------------------

plt.figure()

plt.plot(temperature, energy, marker="o")

plt.xlabel("Temperature (K)")
plt.ylabel("Energy (eV/atom)")
plt.title("Energy versus Temperature")

plt.tight_layout()

plt.savefig("energy_vs_temperature.png", dpi=200)

plt.show()


# ---------------------------------------------------------------------------
# 3. MAKE A SCATTER PLOT
# ---------------------------------------------------------------------------

density = np.array([2.70, 7.87, 8.90, 8.96])
melting_temperature = np.array([933, 1811, 1728, 1358])

plt.figure()

plt.scatter(density, melting_temperature)

plt.xlabel("Density (g/cm^3)")
plt.ylabel("Melting Temperature (K)")
plt.title("Density versus Melting Temperature")

plt.tight_layout()

plt.savefig("density_vs_melting_temperature.png", dpi=200)

plt.show()


# ---------------------------------------------------------------------------
# PRACTICE EXERCISE
# ---------------------------------------------------------------------------

# Create a plot of elastic modulus versus alloy composition.
#
# The code is provided below. Change the values and rerun the script to see
# how the figure changes.

composition = np.array([0, 25, 50, 75, 100])
elastic_modulus = np.array([70, 95, 120, 145, 170])

plt.figure()

plt.plot(composition, elastic_modulus, marker="o")

plt.xlabel("Composition of Element B (%)")
plt.ylabel("Elastic Modulus (GPa)")
plt.title("Elastic Modulus versus Composition")

plt.tight_layout()

plt.savefig("elastic_modulus_vs_composition.png", dpi=200)

plt.show()
