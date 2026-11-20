#!/usr/bin/env python3
"""
Question 4A: Build initial and final vacancy-migration structures in FCC Al.

STUDENT TASKS
-------------
1. Build a 3 x 3 x 3 conventional FCC Al supercell with ASE.
2. Remove one atom near the center of the cell to create a vacancy.
3. Identify one nearest-neighbor Al atom next to the vacancy.
4. Create the final state by moving that Al atom into the original vacancy site.
5. Save the initial and final states.

The supplied helper functions and checks are intended to keep the exercise
focused on constructing the physical migration event rather than bookkeeping.
"""

from pathlib import Path
import json
import numpy as np

from ase.build import bulk
from ase.geometry import find_mic
from ase.io import write

ROOT = Path(__file__).resolve().parent
ENDPOINT_DIR = ROOT / "endpoints"
ENDPOINT_DIR.mkdir(exist_ok=True)

LATTICE_CONSTANT_A = 4.05
REPLICATION = (3, 3, 3)
EXPECTED_BULK_ATOMS = 108
EXPECTED_DEFECT_ATOMS = 107


def atom_closest_to_point(atoms, point):
    """Return index of atom closest to point using minimum-image distances."""
    dr = atoms.positions - np.asarray(point)
    _, distances = find_mic(dr, cell=atoms.cell, pbc=atoms.pbc)
    return int(np.argmin(distances))


def nearest_atom_to_point(atoms, point):
    """Return index and distance of the atom nearest to a point."""
    dr = atoms.positions - np.asarray(point)
    _, distances = find_mic(dr, cell=atoms.cell, pbc=atoms.pbc)
    idx = int(np.argmin(distances))
    return idx, float(distances[idx])


# ============================================================================
# TODO 1 — BUILD PURE FCC Al
# ============================================================================
# Build a conventional cubic FCC Al cell with ASE using:
#   element: Al
#   structure: fcc
#   lattice constant: LATTICE_CONSTANT_A
#   cubic=True
# Then repeat it using REPLICATION.
#
# Store the final 3 x 3 x 3 supercell in the variable:
#     bulk_al
#
# Expected number of atoms: 108
# ============================================================================

#https://docs.ase-lib.org/ase/build/build.html#common-bulk-crystals the command.  build.bulk for crystalls
#grep -rn --include=*.py "bulk(" . | grep -v Homework4
#./TST_and_KMC/Workflow_day/Dimer_neb_example/dimer_neb_oxygen_in_al_demo.py:150:    base = bulk("Al", "fcc", a=A0, cubic=True).repeat(SUPERCELL)
#A0 = 4.05
SUPERCELL = (3, 3, 3)       # 32 Al + 1 O: intentionally small for class
bulk_al = bulk("Al", "fcc", a=LATTICE_CONSTANT_A, cubic=True).repeat(REPLICATION)


if bulk_al is None:
    raise NotImplementedError("TODO 1: build and replicate the FCC Al structure.")

if len(bulk_al) != EXPECTED_BULK_ATOMS:
    raise RuntimeError(
        f"Expected {EXPECTED_BULK_ATOMS} atoms in the perfect crystal, "
        f"but found {len(bulk_al)}."
    )

bulk_al.pbc = True
write(ENDPOINT_DIR / "perfect_al.xyz", bulk_al)


# ============================================================================
# TODO 2 — CREATE ONE VACANCY
# ============================================================================
# Choose an atom near the center of the supercell and remove it.
#
# Helpful quantities already provided:
#   cell_center = 0.5 * sum of the three cell vectors
#   atom_closest_to_point(...)
#
# Requirements:
#   * Store the position of the removed atom in vacancy_position_A.
#   * Store the defective structure in initial.
#   * initial must contain 107 atoms.
# ============================================================================
cell_center = 0.5 * np.sum(bulk_al.cell.array, axis=0)
#print(cell_center)
#breakpoint()
vacancy_index = atom_closest_to_point(bulk_al,cell_center)
vacancy_position_A = bulk_al[vacancy_index].position
initial = bulk_al.copy()
#breakpoint()
del(initial[vacancy_index])


#breakpoint()
if initial is None or vacancy_position_A is None:
    raise NotImplementedError("TODO 2: remove one central Al atom to create a vacancy.")

if len(initial) != EXPECTED_DEFECT_ATOMS:
    raise RuntimeError(
        f"Expected {EXPECTED_DEFECT_ATOMS} atoms after making one vacancy, "
        f"but found {len(initial)}."
    )


# ============================================================================
# TODO 3 — CREATE THE FINAL VACANCY STATE
# ============================================================================
# Find one nearest-neighbor Al atom next to the vacancy and make a copy of
# the initial structure. Move ONLY that atom into vacancy_position_A.
#
# Helpful function:
#   nearest_atom_to_point(initial, vacancy_position_A)
#
# Requirements:
#   * Store the chosen atom index (in the defective structure) in
#       migrating_atom_index
#   * Store the final structure in
#       final
#
# The old site of that migrating atom is now the vacancy in the final state.
# ============================================================================
migrating_atom_index, _ = nearest_atom_to_point(initial, vacancy_position_A)
#breakpoint()
final = initial.copy()
final.positions[migrating_atom_index] = vacancy_position_A

##ase gui endpoints/initial.traj endpoints/final.traj
##two fingers to drag.  52 moves!!

if final is None or migrating_atom_index is None:
    raise NotImplementedError(
        "TODO 3: move one nearest-neighbor Al atom into the vacancy site."
    )

if len(final) != len(initial):
    raise RuntimeError("Initial and final structures must have the same atom count.")

# Validate that exactly one atom moved between the unrelaxed endpoints.
delta = final.positions - initial.positions
mic_delta, move_distances = find_mic(delta, cell=initial.cell, pbc=initial.pbc)
moved = np.where(move_distances > 1.0e-6)[0]

if len(moved) != 1:
    raise RuntimeError(
        f"Exactly one atom should move between the endpoints; found {len(moved)}."
    )

jump_distance_A = float(move_distances[moved[0]])
expected_nn_A = LATTICE_CONSTANT_A / np.sqrt(2.0)

if not np.isclose(jump_distance_A, expected_nn_A, rtol=0.08):
    raise RuntimeError(
        f"The migration jump is {jump_distance_A:.3f} A, but an FCC nearest-"
        f"neighbor jump should be about {expected_nn_A:.3f} A."
    )

# Save endpoints in ASE trajectory format for the NEB script.
write(ENDPOINT_DIR / "initial.traj", initial)
write(ENDPOINT_DIR / "final.traj", final)
write(ENDPOINT_DIR / "initial.xyz", initial)
write(ENDPOINT_DIR / "final.xyz", final)

metadata = {
    "lattice_constant_A": LATTICE_CONSTANT_A,
    "replication": list(REPLICATION),
    "perfect_atom_count": len(bulk_al),
    "defective_atom_count": len(initial),
    "vacancy_index_in_perfect_crystal": int(vacancy_index),
    "vacancy_position_A": np.asarray(vacancy_position_A).tolist(),
    "migrating_atom_index_in_defective_structure": int(migrating_atom_index),
    "jump_distance_A": jump_distance_A,
    "expected_fcc_nearest_neighbor_distance_A": expected_nn_A,
}

with open(ENDPOINT_DIR / "endpoint_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\nVacancy migration endpoints created successfully")
print("================================================")
print(f"Perfect crystal atoms: {len(bulk_al)}")
print(f"Defective atoms:       {len(initial)}")
print(f"Vacancy position:      {np.asarray(vacancy_position_A)} A")
print(f"Migrating atom index:  {migrating_atom_index}")
print(f"Jump distance:         {jump_distance_A:.4f} A")
print("\nCreated:")
print("  endpoints/initial.traj")
print("  endpoints/final.traj")
print("  endpoints/initial.xyz")
print("  endpoints/final.xyz")
print("  endpoints/endpoint_metadata.json")
