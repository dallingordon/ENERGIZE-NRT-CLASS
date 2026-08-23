"""
verify_local_environment.py

Run this script after creating the course conda environment.

The script checks that the main Python packages used in the course can be
imported successfully. It also performs one very small PyTorch calculation
on the CPU.

Run with:

    python verify_local_environment.py
"""

from importlib.metadata import version

print("=" * 70)
print("Intro to Materials by Design: local environment check")
print("=" * 70)
print()

# ---------------------------------------------------------------------------
# 1. IMPORT THE MAIN PACKAGES
# ---------------------------------------------------------------------------

# If one of these imports fails, Python will display an error telling you
# which package could not be found.

import numpy as np
import scipy
import pandas as pd
import matplotlib
import sklearn
import ase
import pymatgen
import networkx as nx
import h5py
import openpyxl
import yaml
import tqdm
import torch

print("All required packages imported successfully.")
print()


# ---------------------------------------------------------------------------
# 2. PRINT PACKAGE VERSIONS
# ---------------------------------------------------------------------------

# Printing versions is useful when troubleshooting because it tells us
# exactly which software is installed in the environment.

packages = [
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "scikit-learn",
    "ase",
    "pymatgen",
    "networkx",
    "h5py",
    "openpyxl",
    "pyyaml",
    "tqdm",
    "torch",
]

print("Installed package versions:")

for package_name in packages:
    print(f"  {package_name:15s} {version(package_name)}")

print()


# ---------------------------------------------------------------------------
# 3. TEST A SMALL NUMPY CALCULATION
# ---------------------------------------------------------------------------

values = np.array([1.0, 2.0, 3.0])
average_value = np.mean(values)

print("NumPy test:")
print("  Values:", values)
print("  Average:", average_value)
print()


# ---------------------------------------------------------------------------
# 4. TEST A SMALL PANDAS DATAFRAME
# ---------------------------------------------------------------------------

data = {
    "Material": ["Al", "Cu", "Ni"],
    "Density": [2.70, 8.96, 8.90],
}

df = pd.DataFrame(data)

print("pandas test:")
print(df)
print()


# ---------------------------------------------------------------------------
# 5. TEST PYTORCH ON THE CPU
# ---------------------------------------------------------------------------

# We explicitly place the tensor on the CPU because the local course
# environment does not require GPU acceleration.

device = torch.device("cpu")

tensor = torch.tensor([1.0, 2.0, 3.0], device=device)
result = tensor * 2.0

print("PyTorch CPU test:")
print("  Device:", tensor.device)
print("  Input: ", tensor)
print("  Result:", result)
print()


# ---------------------------------------------------------------------------
# 6. FINAL MESSAGE
# ---------------------------------------------------------------------------

print("=" * 70)
print("Environment check completed successfully.")
print("=" * 70)
