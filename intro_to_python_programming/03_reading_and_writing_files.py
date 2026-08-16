"""
03_reading_and_writing_files.py

Purpose
-------
This script introduces:
    1. Writing a text file
    2. Reading a text file
    3. Writing a simple CSV file
    4. Using the with statement

Files are important because simulation and experimental data are usually
stored on disk rather than typed directly into a Python script.
"""

# ---------------------------------------------------------------------------
# 1. WRITING A TEXT FILE
# ---------------------------------------------------------------------------

# "w" means write mode.
#
# The with statement automatically closes the file when we are finished.

with open("example_output.txt", "w") as file:
    file.write("Material: Aluminum\n")
    file.write("Temperature: 300 K\n")
    file.write("Energy: -3.25 eV/atom\n")

print("Created example_output.txt")

print()


# ---------------------------------------------------------------------------
# 2. READING A TEXT FILE
# ---------------------------------------------------------------------------

with open("example_output.txt", "r") as file:
    contents = file.read()

print("Contents of example_output.txt:")
print(contents)


# ---------------------------------------------------------------------------
# 3. WRITING SEVERAL VALUES USING A LOOP
# ---------------------------------------------------------------------------

temperatures = [300, 600, 900, 1200]
energies = [-3.30, -3.10, -2.80, -2.40]

with open("temperature_energy.csv", "w") as file:

    # Write a header row first.
    file.write("Temperature_K,Energy_eV\n")

    # zip() lets us step through two lists at the same time.
    for temperature, energy in zip(temperatures, energies):
        file.write(f"{temperature},{energy}\n")

print("Created temperature_energy.csv")

print()


# ---------------------------------------------------------------------------
# 4. READING THE CSV FILE AS PLAIN TEXT
# ---------------------------------------------------------------------------

with open("temperature_energy.csv", "r") as file:
    contents = file.read()

print("Contents of temperature_energy.csv:")
print(contents)


# ---------------------------------------------------------------------------
# PRACTICE EXERCISE
# ---------------------------------------------------------------------------

# Create a file called pressures.txt containing the values:
#
# 1
# 5
# 10
# 20
#
# One value should appear on each line.

pressures = [1, 5, 10, 20]

with open("pressures.txt", "w") as file:

    for pressure in pressures:
        file.write(f"{pressure}\n")

print("Practice exercise:")
print("Created pressures.txt")
