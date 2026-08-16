"""
01_logic_and_loops.py

Purpose
-------
This script introduces:
    1. if statements
    2. Comparison operators
    3. for loops
    4. while loops
    5. Combining logic with loops

These tools allow a program to make decisions and repeat tasks.
"""

# ---------------------------------------------------------------------------
# 1. IF STATEMENTS
# ---------------------------------------------------------------------------

# An if statement allows Python to make a decision.

temperature = 1200

if temperature > 1000:
    print("The temperature is above 1000 K.")

print()


# ---------------------------------------------------------------------------
# 2. IF / ELSE
# ---------------------------------------------------------------------------

temperature = 500

if temperature > 1000:
    print("High-temperature simulation")
else:
    print("Low-temperature simulation")

print()


# ---------------------------------------------------------------------------
# 3. IF / ELIF / ELSE
# ---------------------------------------------------------------------------

temperature = 900

if temperature < 500:
    print("Temperature range: low")
elif temperature < 1000:
    print("Temperature range: medium")
else:
    print("Temperature range: high")

print()


# ---------------------------------------------------------------------------
# 4. COMPARISON OPERATORS
# ---------------------------------------------------------------------------

# Common comparison operators:
#
# ==    equal to
# !=    not equal to
# >     greater than
# <     less than
# >=    greater than or equal to
# <=    less than or equal to

pressure = 1.0

if pressure == 1.0:
    print("The pressure is exactly 1 atm.")

print()


# ---------------------------------------------------------------------------
# 5. FOR LOOPS
# ---------------------------------------------------------------------------

# A for loop repeats a block of code.
#
# range(5) produces:
# 0, 1, 2, 3, 4

print("Loop example:")

for i in range(5):
    print("Step number:", i)

print()


# ---------------------------------------------------------------------------
# 6. LOOPING THROUGH A LIST
# ---------------------------------------------------------------------------

temperatures = [300, 600, 900, 1200]

for temperature in temperatures:
    print("Running simulation at", temperature, "K")

print()


# ---------------------------------------------------------------------------
# 7. LOGIC INSIDE A LOOP
# ---------------------------------------------------------------------------

energies = [-3.1, -2.8, -1.5, -0.4]

for energy in energies:

    if energy < -2.0:
        print(energy, "eV -> low energy")
    else:
        print(energy, "eV -> high energy")

print()


# ---------------------------------------------------------------------------
# 8. WHILE LOOPS
# ---------------------------------------------------------------------------

# A while loop continues until a condition is no longer true.

step = 0

while step < 5:
    print("Current step:", step)
    step = step + 1

print()


# ---------------------------------------------------------------------------
# PRACTICE EXERCISE
# ---------------------------------------------------------------------------

# The temperatures below represent several molecular dynamics simulations.
#
# Print each temperature and classify it as:
#
# "solid-like" if temperature is less than 900 K
# "high-temperature" if temperature is 900 K or greater

temperatures = [300, 600, 900, 1200, 1500]

print("Practice exercise:")

for temperature in temperatures:

    if temperature < 900:
        print(temperature, "K -> solid-like")
    else:
        print(temperature, "K -> high-temperature")
