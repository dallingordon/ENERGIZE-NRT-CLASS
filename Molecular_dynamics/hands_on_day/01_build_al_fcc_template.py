#!/usr/bin/env python3
"""
IN-CLASS ACTIVITY: Build FCC Al for LAMMPS using ASE

Complete three tasks:
1. Build conventional FCC Al with a = 4.05 A.
2. Replicate it 4 x 4 x 4.
3. Write al_fcc_4x4x4.data in LAMMPS-data format.

Expected atoms: 256
"""

from pathlib import Path
from ase.build import bulk
from ase.io import write

ROOT = Path(__file__).resolve().parent
LATTICE_CONSTANT_A = 4.05
REPLICATION = (4, 4, 4)
OUTPUT_FILE = ROOT / "al_fcc_4x4x4.data"

# ============================================================
# TODO 1 — Build the conventional cubic FCC Al unit cell.
# Use ASE bulk() with:
#   element = Al
#   structure = fcc
#   a = LATTICE_CONSTANT_A
#   conventional cubic cell
# The variable must be named atoms.
# ============================================================
atoms = None

# ============================================================
# TODO 2 — Replicate the structure using REPLICATION.
# You may use atoms.repeat(...) or equivalent ASE syntax.
# ============================================================


if atoms is None:
    raise NotImplementedError("TODO: build the FCC Al structure with ASE.")

if len(atoms) == 4:
    raise RuntimeError("Replicate the conventional FCC cell 4 x 4 x 4.")

if len(atoms) != 256:
    raise RuntimeError(f"Expected 256 atoms, but found {len(atoms)}.")

# ============================================================
# TODO 3 — Write OUTPUT_FILE using ASE write().
# Use:
#   format="lammps-data"
#   atom_style="atomic"
# ============================================================


print("\nStructure check")
print("===============")
print(f"Atoms: {len(atoms)}")
print("Cell (A):")
print(atoms.cell)
print(f"\nExpected output file: {OUTPUT_FILE}")

if not OUTPUT_FILE.exists():
    raise FileNotFoundError(
        "LAMMPS data file not found. Complete TODO 3 using ASE write()."
    )

print("\nSuccess: LAMMPS data file created.")
