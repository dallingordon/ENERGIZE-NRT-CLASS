"""
09_structured_material_data.py

Purpose
-------
This example shows how raw data from many separate material files can be
organized into one clean pandas DataFrame.

The script performs four main steps:

    1. Generate synthetic but physically reasonable materials data.
    2. Save one CSV file for each material.
    3. Read all of the individual files into pandas.
    4. Combine the data and save one master dataset.

The resulting master dataset is saved as both:

    material_properties.csv
    material_properties.xlsx

CSV files are especially convenient for future Python scripts because they
can be loaded with:

    df = pd.read_csv("material_properties.csv")

The Excel file contains the same information and can be opened directly
with Microsoft Excel.

IMPORTANT
---------
The values in this example are synthetic. They are not measurements or
simulation results for real named materials.

The values are generated so that they behave approximately like properties
of crystalline metallic materials. For example, materials with stronger
average bonding tend to have:

    - larger vacancy formation energies
    - larger bulk moduli
    - larger surface energies
    - higher melting temperatures

Stacking fault energy is given a weaker relationship to bond strength because
it depends strongly on crystal structure and the detailed defect energetics.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. USER SETTINGS
# ---------------------------------------------------------------------------

# Number of synthetic materials to create.
NUMBER_OF_MATERIALS = 100

# A random seed makes the generated dataset reproducible.
#
# If every student uses the same seed, everyone will generate the same data.
RANDOM_SEED = 42

# Folder containing one raw CSV file for each material.
RAW_DATA_DIRECTORY = Path("raw_material_files")

# Folder containing the final combined datasets.
PROCESSED_DATA_DIRECTORY = Path("processed_data")


# ---------------------------------------------------------------------------
# 2. CREATE THE OUTPUT DIRECTORIES
# ---------------------------------------------------------------------------

# exist_ok=True means Python will not produce an error if the folder already
# exists.

RAW_DATA_DIRECTORY.mkdir(exist_ok=True)
PROCESSED_DATA_DIRECTORY.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 3. SET UP THE RANDOM NUMBER GENERATOR
# ---------------------------------------------------------------------------

rng = np.random.default_rng(RANDOM_SEED)


# ---------------------------------------------------------------------------
# 4. GENERATE ONE SYNTHETIC MATERIAL AT A TIME
# ---------------------------------------------------------------------------

# We will create several related properties.
#
# Instead of generating every property completely independently, we first
# create a hidden "bonding strength" variable.
#
# This is useful because real materials properties are often correlated.
# Stronger bonding usually makes it harder to remove atoms or separate
# surfaces, and often increases stiffness and melting temperature.

for material_number in range(1, NUMBER_OF_MATERIALS + 1):

    # Create an easy-to-read ID such as:
    #
    # MAT_001
    # MAT_002
    # MAT_003
    #
    material_id = f"MAT_{material_number:03d}"

    # -----------------------------------------------------------------------
    # Average bond strength
    # -----------------------------------------------------------------------

    # Typical metallic bonding energies vary substantially.
    #
    # Here we generate values between approximately 1.3 and 4.5 eV/bond.
    # Most generated materials will fall near the middle of this range.

    average_bond_strength = rng.normal(loc=2.7, scale=0.65)

    # np.clip() prevents unrealistic values from being generated.
    average_bond_strength = np.clip(
        average_bond_strength,
        1.3,
        4.5
    )

    # -----------------------------------------------------------------------
    # Vacancy formation energy
    # -----------------------------------------------------------------------

    # Stronger bonding generally makes it more difficult to remove an atom.
    #
    # The random term prevents the relationship from being perfectly linear.

    vacancy_formation_energy = (
        0.35
        + 0.70 * average_bond_strength
        + rng.normal(loc=0.0, scale=0.18)
    )

    vacancy_formation_energy = np.clip(
        vacancy_formation_energy,
        0.8,
        3.8
    )

    # -----------------------------------------------------------------------
    # Bulk modulus
    # -----------------------------------------------------------------------

    # Bulk modulus describes resistance to uniform compression.
    #
    # Stronger-bonded metallic materials often have larger bulk moduli.

    bulk_modulus = (
        20.0
        + 52.0 * average_bond_strength
        + rng.normal(loc=0.0, scale=15.0)
    )

    bulk_modulus = np.clip(
        bulk_modulus,
        45.0,
        285.0
    )

    # -----------------------------------------------------------------------
    # Average surface energy
    # -----------------------------------------------------------------------

    # Creating a surface requires breaking or weakening bonds.
    # Therefore, surface energy is also related to bond strength.

    average_surface_energy = (
        0.30
        + 0.68 * average_bond_strength
        + rng.normal(loc=0.0, scale=0.20)
    )

    average_surface_energy = np.clip(
        average_surface_energy,
        0.8,
        3.6
    )

    # -----------------------------------------------------------------------
    # Stacking fault energy
    # -----------------------------------------------------------------------

    # Stacking fault energy is more complicated.
    #
    # It depends on bonding, but also strongly on crystal structure and local
    # atomic arrangements.
    #
    # We therefore create a second hidden variable so that stacking fault
    # energy is only weakly correlated with average bond strength.

    defect_character = rng.uniform(0.0, 1.0)

    stacking_fault_energy = (
        20.0
        + 28.0 * average_bond_strength
        + 185.0 * defect_character
        + rng.normal(loc=0.0, scale=20.0)
    )

    stacking_fault_energy = np.clip(
        stacking_fault_energy,
        25.0,
        350.0
    )

    # -----------------------------------------------------------------------
    # Melting temperature
    # -----------------------------------------------------------------------

    # Stronger atomic bonding generally requires more thermal energy to
    # destroy the crystalline structure.

    melting_temperature = (
        380.0
        + 470.0 * average_bond_strength
        + rng.normal(loc=0.0, scale=130.0)
    )

    melting_temperature = np.clip(
        melting_temperature,
        700.0,
        2800.0
    )

    # -----------------------------------------------------------------------
    # Put the values into a one-row pandas DataFrame
    # -----------------------------------------------------------------------

    material_data = {
        "Material_ID": [material_id],
        "Vacancy_Formation_Energy_eV": [vacancy_formation_energy],
        "Bulk_Modulus_GPa": [bulk_modulus],
        "Average_Surface_Energy_J_m2": [average_surface_energy],
        "Stacking_Fault_Energy_mJ_m2": [stacking_fault_energy],
        "Average_Bond_Strength_eV": [average_bond_strength],
        "Melting_Temperature_K": [melting_temperature],
    }

    material_df = pd.DataFrame(material_data)

    # Round the values so the raw files are easy to read.
    material_df = material_df.round(3)

    # -----------------------------------------------------------------------
    # Save one file for this material
    # -----------------------------------------------------------------------

    output_file = RAW_DATA_DIRECTORY / f"{material_id}.csv"

    material_df.to_csv(
        output_file,
        index=False
    )


print(f"Generated {NUMBER_OF_MATERIALS} raw material files.")
print()


# ---------------------------------------------------------------------------
# 5. FIND ALL OF THE RAW MATERIAL FILES
# ---------------------------------------------------------------------------

# glob("MAT_*.csv") finds every CSV file whose name starts with MAT_.
#
# sorted() puts them into a predictable numerical order.

raw_files = sorted(RAW_DATA_DIRECTORY.glob("MAT_*.csv"))

print("Number of raw files found:", len(raw_files))
print()


# ---------------------------------------------------------------------------
# 6. READ EACH FILE INTO PANDAS
# ---------------------------------------------------------------------------

# We will store each small one-row DataFrame in this list.

all_material_dataframes = []

for raw_file in raw_files:

    # Read one material file.
    material_df = pd.read_csv(raw_file)

    # Add the DataFrame to our list.
    all_material_dataframes.append(material_df)


# ---------------------------------------------------------------------------
# 7. COMBINE ALL MATERIALS INTO ONE DATAFRAME
# ---------------------------------------------------------------------------

# pd.concat() joins all of the one-row DataFrames together.
#
# ignore_index=True creates a new row index:
#
# 0, 1, 2, 3, ...

master_df = pd.concat(
    all_material_dataframes,
    ignore_index=True
)


# ---------------------------------------------------------------------------
# 8. LOOK AT THE COMBINED DATA
# ---------------------------------------------------------------------------

print("First five rows of the combined DataFrame:")
print(master_df.head())
print()

print("DataFrame shape:")
print(master_df.shape)
print()

print("Column names:")
print(master_df.columns.tolist())
print()


# ---------------------------------------------------------------------------
# 9. SAVE THE MASTER DATASET AS A CSV FILE
# ---------------------------------------------------------------------------

csv_output_file = (
    PROCESSED_DATA_DIRECTORY
    / "material_properties.csv"
)

master_df.to_csv(
    csv_output_file,
    index=False
)

print("Saved:")
print(csv_output_file)
print()


# ---------------------------------------------------------------------------
# 10. SAVE THE SAME DATASET AS AN EXCEL FILE
# ---------------------------------------------------------------------------

excel_output_file = (
    PROCESSED_DATA_DIRECTORY
    / "material_properties.xlsx"
)

master_df.to_excel(
    excel_output_file,
    index=False
)

print("Saved:")
print(excel_output_file)
print()


# ---------------------------------------------------------------------------
# 11. SHOW HOW THE DATA CAN BE READ AGAIN LATER
# ---------------------------------------------------------------------------

# This is the most important line for future exercises.
#
# Any later Python script can load the complete dataset with one command.

loaded_df = pd.read_csv(csv_output_file)

print("Dataset loaded back from the master CSV file:")
print(loaded_df.head())
print()


# ---------------------------------------------------------------------------
# 12. SIMPLE EXAMPLES OF USING THE DATAFRAME
# ---------------------------------------------------------------------------

# Example 1:
# Find the material with the largest bulk modulus.

largest_bulk_modulus_row = loaded_df.loc[
    loaded_df["Bulk_Modulus_GPa"].idxmax()
]

print("Material with the largest bulk modulus:")
print(largest_bulk_modulus_row)
print()


# Example 2:
# Find materials with melting temperatures greater than 1800 K.

high_melting_materials = loaded_df[
    loaded_df["Melting_Temperature_K"] > 1800
]

print("Materials with melting temperatures above 1800 K:")
print(high_melting_materials)
print()


# ---------------------------------------------------------------------------
# PRACTICE EXERCISE
# ---------------------------------------------------------------------------

# Try these on your own:
#
# 1. Find all materials with vacancy formation energy above 2.0 eV.
#
# 2. Find the material with the lowest average surface energy.
#
# 3. Calculate the average bulk modulus of all 100 materials.
#
# 4. Sort the DataFrame from highest to lowest melting temperature.
#
# 5. Save the sorted DataFrame as:
#
#       materials_sorted_by_melting_temperature.csv
#
#
# Example solution for Question 3:

average_bulk_modulus = loaded_df["Bulk_Modulus_GPa"].mean()

print("Practice example:")
print("Average bulk modulus =", average_bulk_modulus, "GPa")
