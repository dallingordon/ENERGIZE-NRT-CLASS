"""
02_lists_dictionaries_and_functions.py

Purpose
-------
This script introduces:
    1. Lists
    2. Dictionaries
    3. Accessing stored values
    4. Functions

Lists and dictionaries are simple ways to organize information.
Functions help us reuse code.
"""

# ---------------------------------------------------------------------------
# 1. LISTS
# ---------------------------------------------------------------------------

# A list stores multiple values in one variable.

temperatures = [300, 600, 900, 1200]

print("Temperatures:")
print(temperatures)

print()


# ---------------------------------------------------------------------------
# 2. ACCESSING ITEMS IN A LIST
# ---------------------------------------------------------------------------

# Python starts counting positions at 0.

print("First temperature:", temperatures[0])
print("Second temperature:", temperatures[1])

print()


# ---------------------------------------------------------------------------
# 3. ADDING TO A LIST
# ---------------------------------------------------------------------------

temperatures.append(1500)

print("Updated temperatures:")
print(temperatures)

print()


# ---------------------------------------------------------------------------
# 4. DICTIONARIES
# ---------------------------------------------------------------------------

# A dictionary stores information as key:value pairs.

material = {
    "name": "Aluminum",
    "symbol": "Al",
    "atomic_number": 13,
    "lattice_constant": 4.05
}

print("Material dictionary:")
print(material)

print()


# We can access an individual value using its key.

print("Material name:", material["name"])
print("Chemical symbol:", material["symbol"])
print("Lattice constant:", material["lattice_constant"], "Angstrom")

print()


# ---------------------------------------------------------------------------
# 5. FUNCTIONS
# ---------------------------------------------------------------------------

# A function is a reusable block of code.
#
# The function below converts temperature from Celsius to Kelvin.

def celsius_to_kelvin(celsius_temperature):
    kelvin_temperature = celsius_temperature + 273.15
    return kelvin_temperature


room_temperature_c = 25.0
room_temperature_k = celsius_to_kelvin(room_temperature_c)

print("Room temperature:")
print(room_temperature_c, "degrees C")
print(room_temperature_k, "K")

print()


# ---------------------------------------------------------------------------
# 6. A FUNCTION WITH TWO INPUTS
# ---------------------------------------------------------------------------

def energy_per_atom(total_energy, number_of_atoms):
    result = total_energy / number_of_atoms
    return result


value = energy_per_atom(-845.0, 256)

print("Energy per atom:")
print(value, "eV/atom")

print()


# ---------------------------------------------------------------------------
# PRACTICE EXERCISE
# ---------------------------------------------------------------------------

# Write a function that calculates density:
#
# density = mass / volume
#
# The example below uses:
# mass = 27.0 g
# volume = 10.0 cm^3

def calculate_density(mass, volume):
    density = mass / volume
    return density


mass = 27.0
volume = 10.0

density = calculate_density(mass, volume)

print("Practice exercise:")
print("Density =", density, "g/cm^3")
