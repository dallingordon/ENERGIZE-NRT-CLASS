#!/usr/bin/env python3
"""01 - Build and relax a nearest-neighbor Al divacancy.

STUDENT TASKS:
  TODO 1: build a 3x3x3 conventional FCC Al crystal with ASE.
  TODO 2: choose two nearest-neighbor lattice sites to remove.

The supplied code handles the EAM relaxation and metadata/output.
"""

from pathlib import Path
import numpy as np
from ase.build import bulk
from ase.constraints import FixAtoms
from ase.io import write
from ase.optimize import FIRE

from common import (
    make_calculator, pair_candidates_near_center, choose_anchor_atom,
    dump_json, mic_distance
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)

A0 = 4.05
REPLICATION = (3, 3, 3)
NN_DISTANCE = A0 / np.sqrt(2.0)

# =====================================================================
# TODO 1 - BUILD PURE FCC Al
# =====================================================================
# Use ase.build.bulk(..., crystalstructure='fcc', a=A0, cubic=True)
# and replicate the conventional cubic cell by REPLICATION.
# The final perfect structure should contain 108 atoms.
#
atoms = None

if atoms is None:
    raise NotImplementedError('TODO 1: build the 3x3x3 conventional FCC Al crystal.')
if len(atoms) != 108:
    raise RuntimeError(f'Expected 108 atoms, found {len(atoms)}.')

write(OUT / 'perfect_fcc.traj', atoms)

# Print reasonable nearest-neighbor pairs near the center of the cell.
candidates = pair_candidates_near_center(atoms, NN_DISTANCE)
print('\nCandidate nearest-neighbor vacancy pairs near the cell center:')
print(' pair   atom_i  atom_j   separation(A)')
for k, (_, d, i, j, _) in enumerate(candidates):
    print(f' {k:>4d}   {i:>6d}  {j:>6d}      {d:8.4f}')

# =====================================================================
# TODO 2 - CHOOSE THE TWO LATTICE SITES TO REMOVE
# =====================================================================
# Pick one nearest-neighbor pair from the table above and enter the two
# atom indices, for example: VACANCY_PAIR = (12, 15)
#
VACANCY_PAIR = None

if VACANCY_PAIR is None:
    raise NotImplementedError('TODO 2: choose a nearest-neighbor pair to remove.')

i, j = map(int, VACANCY_PAIR)
if i == j or i < 0 or j < 0 or i >= len(atoms) or j >= len(atoms):
    raise ValueError('VACANCY_PAIR contains invalid atom indices.')

pair_distance = atoms.get_distance(i, j, mic=True)
if abs(pair_distance - NN_DISTANCE) > 0.15:
    raise RuntimeError(
        f'The selected sites are {pair_distance:.3f} A apart; expected a nearest-neighbor '
        f'distance near {NN_DISTANCE:.3f} A.'
    )

vacancy_positions = [atoms.positions[i].copy(), atoms.positions[j].copy()]
defect_center = vacancy_positions[0] + 0.5 * (
    vacancy_positions[1] - vacancy_positions[0]
)

# Delete in descending index order so both requested atoms are removed correctly.
defect = atoms.copy()
for idx in sorted([i, j], reverse=True):
    del defect[idx]

if len(defect) != 106:
    raise RuntimeError(f'Expected 106 atoms after creating a divacancy, found {len(defect)}.')

# Pin one remote atom only to remove the global translational zero mode.
anchor_index = choose_anchor_atom(defect, defect_center)
defect.set_constraint(FixAtoms(indices=[anchor_index]))

defect.calc = make_calculator()
initial_energy = defect.get_potential_energy()

opt = FIRE(defect, logfile=str(OUT / 'divacancy_relax.log'), trajectory=str(OUT / 'divacancy_relax.traj'))
opt.run(fmax=0.02, steps=1000)
final_energy = defect.get_potential_energy()
write(OUT / 'divacancy_relaxed.traj', defect)

meta = {
    'lattice_constant_A': A0,
    'replication': list(REPLICATION),
    'perfect_atoms': 108,
    'defect_atoms': len(defect),
    'removed_original_indices': [i, j],
    'vacancy_positions_A': [p.tolist() for p in vacancy_positions],
    'initial_vacancy_separation_A': float(pair_distance),
    'anchor_index_in_defect': int(anchor_index),
    'unrelaxed_energy_eV': float(initial_energy),
    'relaxed_energy_eV': float(final_energy),
}
dump_json(OUT / 'divacancy_metadata.json', meta)

print('\nDivacancy created and relaxed successfully.')
print(f'Atoms remaining: {len(defect)}')
print(f'Initial vacancy separation: {pair_distance:.4f} A')
print(f'Relaxed energy: {final_energy:.6f} eV')
print(f'Pinned remote atom index: {anchor_index}')
print('Wrote results/divacancy_relaxed.traj')
