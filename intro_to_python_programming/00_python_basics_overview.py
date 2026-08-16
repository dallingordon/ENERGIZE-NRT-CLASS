"""
00_python_basics_overview.py

Purpose
-------
This script introduces a few of the most basic Python ideas:
    1. Printing information to the screen
    2. Creating variables
    3. Working with numbers and text
    4. Doing simple calculations
    5. Checking the type of a variable

Run this script first.
"""

# ---------------------------------------------------------------------------
# 1. PRINTING TEXT
# ---------------------------------------------------------------------------

# print() displays information in the terminal.
print("Welcome to Intro to Materials by Design!")

# Blank print statements can be used to add space in the output.
print()


# ---------------------------------------------------------------------------
# 2. VARIABLES
# ---------------------------------------------------------------------------

# A variable is a name that stores a value.
# The variable name is on the left side of the equals sign.
# The value being stored is on the right side.

temperature = 300
pressure = 1.0
material_name = "Aluminum"

print("Material:", material_name)
print("Temperature:", temperature, "K")
print("Pressure:", pressure, "atm")

print()


# ---------------------------------------------------------------------------
# 3. BASIC VARIABLE TYPES
# ---------------------------------------------------------------------------

# Python commonly uses several basic data types.

number_of_atoms = 108          # Integer: a whole number
lattice_constant = 4.05        # Float: a decimal number
element = "Al"                 # String: text
simulation_finished = True     # Boolean: True or False

# type() tells us the type of a variable.
print("Variable types:")
print(type(number_of_atoms))
print(type(lattice_constant))
print(type(element))
print(type(simulation_finished))

print()


# ---------------------------------------------------------------------------
# 4. BASIC CALCULATIONS
# ---------------------------------------------------------------------------

# Python can be used like a calculator.

energy_atom_1 = -3.20
energy_atom_2 = -3.35
energy_atom_3 = -3.10

total_energy = energy_atom_1 + energy_atom_2 + energy_atom_3
average_energy = total_energy / 3

print("Total energy =", total_energy, "eV")
print("Average energy =", average_energy, "eV")

print()


# ---------------------------------------------------------------------------
# 5. CHANGING A VARIABLE
# ---------------------------------------------------------------------------

temperature = 300
print("Starting temperature:", temperature, "K")

temperature = temperature + 100
print("New temperature:", temperature, "K")

print()


# ---------------------------------------------------------------------------
# PRACTICE EXERCISE
# ---------------------------------------------------------------------------

# Imagine that a simulation contains 256 atoms and has a total energy
# of -845.0 eV.
#
# Calculate the energy per atom.
#
# Try writing the calculation yourself before looking at the example solution.

number_of_atoms = 256
total_energy = -845.0

# Write your calculation below:
energy_per_atom = total_energy / number_of_atoms

print("Practice exercise:")
print("Energy per atom =", energy_per_atom, "eV/atom")
