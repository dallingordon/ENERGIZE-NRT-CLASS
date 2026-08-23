"""
04_numpy_basics.py

Purpose
-------
This script introduces NumPy.

NumPy is a Python library for numerical calculations. It is especially useful
for arrays, vectors, matrices, and scientific data.

This script introduces:
    1. Creating NumPy arrays
    2. Array indexing
    3. Array calculations
    4. Mean, minimum, and maximum
    5. Simple two-dimensional arrays
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. CREATING A NUMPY ARRAY
# ---------------------------------------------------------------------------

temperatures = np.array([300, 600, 900, 1200])

print("Temperature array:")
print(temperatures)

print()


# ---------------------------------------------------------------------------
# 2. ACCESSING VALUES
# ---------------------------------------------------------------------------

print("First temperature:", temperatures[0])
print("Last temperature:", temperatures[-1])

print()


# ---------------------------------------------------------------------------
# 3. DOING CALCULATIONS ON AN ENTIRE ARRAY
# ---------------------------------------------------------------------------

# NumPy can perform the same operation on every value in an array.

temperatures_celsius = temperatures - 273.15

print("Temperatures in Celsius:")
print(temperatures_celsius)

print()


# ---------------------------------------------------------------------------
# 4. BASIC STATISTICS
# ---------------------------------------------------------------------------

energies = np.array([-3.30, -3.10, -2.80, -2.40])

print("Energy values:")
print(energies)

print("Average energy:", np.mean(energies))
print("Minimum energy:", np.min(energies))
print("Maximum energy:", np.max(energies))

print()


# ---------------------------------------------------------------------------
# 5. TWO-DIMENSIONAL ARRAYS
# ---------------------------------------------------------------------------

# A two-dimensional array can be thought of as a table or matrix.

positions = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0]
])

print("Atomic positions:")
print(positions)

print()


# The shape tells us the number of rows and columns.

print("Array shape:", positions.shape)

print()


# ---------------------------------------------------------------------------
# 6. ACCESSING A ROW
# ---------------------------------------------------------------------------

first_atom_position = positions[0]

print("Position of atom 1:")
print(first_atom_position)

print()


# ---------------------------------------------------------------------------
# PRACTICE EXERCISE
# ---------------------------------------------------------------------------

# The values below are simulated elastic moduli in GPa.
#
# Calculate:
#   1. The average
#   2. The minimum
#   3. The maximum

elastic_moduli = np.array([195.0, 202.0, 198.0, 205.0, 200.0])

average_modulus = np.mean(elastic_moduli)
minimum_modulus = np.min(elastic_moduli)
maximum_modulus = np.max(elastic_moduli)

print("Practice exercise:")
print("Average modulus:", average_modulus, "GPa")
print("Minimum modulus:", minimum_modulus, "GPa")
print("Maximum modulus:", maximum_modulus, "GPa")
