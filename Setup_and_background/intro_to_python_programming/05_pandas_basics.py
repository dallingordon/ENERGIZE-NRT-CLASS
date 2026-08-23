"""
05_pandas_basics.py

Purpose
-------
This script introduces pandas.

pandas is commonly used to work with tables of data.

This script introduces:
    1. Creating a DataFrame
    2. Viewing data
    3. Selecting columns
    4. Filtering rows
    5. Adding a new column
    6. Saving and loading CSV files
"""

import pandas as pd


# ---------------------------------------------------------------------------
# 1. CREATE A DATAFRAME
# ---------------------------------------------------------------------------

# A DataFrame is similar to a spreadsheet.

data = {
    "Material": ["Al", "Cu", "Ni", "Fe"],
    "Density_g_cm3": [2.70, 8.96, 8.90, 7.87],
    "Melting_Temperature_K": [933, 1358, 1728, 1811]
}

df = pd.DataFrame(data)

print("Full DataFrame:")
print(df)

print()


# ---------------------------------------------------------------------------
# 2. SELECT ONE COLUMN
# ---------------------------------------------------------------------------

print("Density column:")
print(df["Density_g_cm3"])

print()


# ---------------------------------------------------------------------------
# 3. BASIC COLUMN STATISTICS
# ---------------------------------------------------------------------------

average_density = df["Density_g_cm3"].mean()

print("Average density:", average_density, "g/cm^3")

print()


# ---------------------------------------------------------------------------
# 4. FILTER ROWS
# ---------------------------------------------------------------------------

# Select materials with melting temperature above 1500 K.

high_melting = df[df["Melting_Temperature_K"] > 1500]

print("Materials with melting temperature above 1500 K:")
print(high_melting)

print()


# ---------------------------------------------------------------------------
# 5. ADD A NEW COLUMN
# ---------------------------------------------------------------------------

# We can calculate a new column from existing data.

df["Density_kg_m3"] = df["Density_g_cm3"] * 1000

print("DataFrame with new density column:")
print(df)

print()


# ---------------------------------------------------------------------------
# 6. SAVE THE DATAFRAME
# ---------------------------------------------------------------------------

df.to_csv("materials_table.csv", index=False)

print("Saved materials_table.csv")

print()


# ---------------------------------------------------------------------------
# 7. READ THE CSV FILE BACK INTO PYTHON
# ---------------------------------------------------------------------------

loaded_df = pd.read_csv("materials_table.csv")

print("Data loaded from materials_table.csv:")
print(loaded_df)

print()


# ---------------------------------------------------------------------------
# PRACTICE EXERCISE
# ---------------------------------------------------------------------------

# Find all materials with density greater than 8.0 g/cm^3.

dense_materials = df[df["Density_g_cm3"] > 8.0]

print("Practice exercise:")
print("Materials with density greater than 8.0 g/cm^3:")
print(dense_materials)
