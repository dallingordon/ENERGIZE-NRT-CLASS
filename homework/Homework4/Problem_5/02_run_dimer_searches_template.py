#!/usr/bin/env python3
"""02 - Run several Dimer searches from the same relaxed Al divacancy.

The script first prints atoms nearest to each vacant lattice site.

STUDENT TASK:
  Choose SIX atom/vacancy combinations and enter them in SEARCH_SPECS.
  Each selected atom will be initially perturbed toward one of the two
  vacancy positions, after which ASE's Dimer method searches for a nearby
  first-order saddle point.
"""

from pathlib import Path
import csv
import numpy as np
from ase.io import read, write
from ase.mep import DimerControl, MinModeAtoms, MinModeTranslate

from common import make_calculator, load_json, normalized

ROOT = Path(__file__).resolve().parent
RES = ROOT / 'results'
DOUT = RES / 'dimer'
DOUT.mkdir(parents=True, exist_ok=True)

base = read(RES / 'divacancy_relaxed.traj')
meta = load_json(RES / 'divacancy_metadata.json')
vacancy_positions = [np.array(p, float) for p in meta['vacancy_positions_A']]
base_energy = float(meta['relaxed_energy_eV'])

from common import nearby_atoms_for_vacancy

print('\nCandidate atoms near vacancy 1:')
print(' rank  atom_index  distance(A)   unit vector atom -> vacancy')
for rank, (d, idx, vec) in enumerate(nearby_atoms_for_vacancy(base, vacancy_positions[0], n=10), 1):
    u = normalized(vec)
    print(f'{rank:>4d}  {idx:>10d}  {d:>10.4f}   [{u[0]: .3f}, {u[1]: .3f}, {u[2]: .3f}]')

print('\nCandidate atoms near vacancy 2:')
print(' rank  atom_index  distance(A)   unit vector atom -> vacancy')
for rank, (d, idx, vec) in enumerate(nearby_atoms_for_vacancy(base, vacancy_positions[1], n=10), 1):
    u = normalized(vec)
    print(f'{rank:>4d}  {idx:>10d}  {d:>10.4f}   [{u[0]: .3f}, {u[1]: .3f}, {u[2]: .3f}]')

# =====================================================================
# TODO - DEFINE SIX LOCAL DIMER SEARCHES
# =====================================================================
# Each dictionary requires:
#   label:       S01, S02, ...
#   atom_index:  an atom index from the tables above
#   vacancy:     1 or 2, telling the script which vacant site that atom
#                should initially move toward
#
# Choose different neighboring atoms/directions so the searches probe
# different possible rearrangements of the divacancy.
#
# Example FORMAT ONLY:
# SEARCH_SPECS = [
#     {'label': 'S01', 'atom_index': 17, 'vacancy': 1},
#     ...
# ]
#
SEARCH_SPECS = []

if len(SEARCH_SPECS) != 6:
    raise NotImplementedError('Define exactly six Dimer searches in SEARCH_SPECS.')

DISPLACEMENT_A = 0.15
FMAX = 0.03
MAX_STEPS = 600

rows = []
for n, spec in enumerate(SEARCH_SPECS, 1):
    label = str(spec['label'])
    atom_index = int(spec['atom_index'])
    vacancy_number = int(spec['vacancy'])
    if vacancy_number not in (1, 2):
        raise ValueError(f'{label}: vacancy must be 1 or 2.')
    if not (0 <= atom_index < len(base)):
        raise ValueError(f'{label}: invalid atom_index {atom_index}.')

    work = base.copy()
    work.calc = make_calculator()
    target = vacancy_positions[vacancy_number - 1]

    # Initial mode: selected atom moves toward the selected vacancy.
    from common import mic_vector
    direction = normalized(mic_vector(work, work.positions[atom_index], target))
    displacement = np.zeros((len(work), 3), float)
    displacement[atom_index] = DISPLACEMENT_A * direction
    mask = [i == atom_index for i in range(len(work))]

    print(f'\n--- {label}: atom {atom_index} toward vacancy {vacancy_number} ---')
    with DimerControl(
        initial_eigenmode_method='displacement',
        displacement_method='vector',
        mask=mask,
        logfile=str(DOUT / f'{label}_dimer.log'),
        eigenmode_logfile=str(DOUT / f'{label}_eigenmode.log'),
        maximum_translation=0.10,
    ) as control:
        d_atoms = MinModeAtoms(work, control, random_seed=1000 + n)
        d_atoms.displace(displacement_vector=displacement)
        with MinModeTranslate(
            d_atoms,
            trajectory=str(DOUT / f'{label}_trajectory.traj'),
            logfile=str(DOUT / f'{label}_translate.log'),
        ) as opt:
            converged = bool(opt.run(fmax=FMAX, steps=MAX_STEPS))

        saddle_energy = float(d_atoms.get_potential_energy())
        curvature = float(d_atoms.get_curvature())
        mode = np.asarray(d_atoms.get_eigenmode(), float)
        saddle_atoms = d_atoms.get_atoms()
        write(DOUT / f'{label}_saddle.traj', saddle_atoms)
        np.save(DOUT / f'{label}_eigenmode.npy', mode)

    barrier = saddle_energy - base_energy
    rows.append({
        'label': label,
        'atom_index': atom_index,
        'vacancy': vacancy_number,
        'converged': converged,
        'curvature_eV_A2': curvature,
        'saddle_energy_eV': saddle_energy,
        'barrier_from_start_eV': barrier,
    })
    print(f'{label}: converged={converged}, curvature={curvature:.6f}, barrier={barrier:.6f} eV')

with (DOUT / 'dimer_search_summary.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print('\nFinished all Dimer searches.')
print('Inspect results/dimer/dimer_search_summary.csv and the saddle structures.')
