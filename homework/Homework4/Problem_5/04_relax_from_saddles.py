#!/usr/bin/env python3
"""04 - Push each selected Dimer saddle downhill in +/- unstable-mode directions.

This script is complete. It discovers the two minima connected by each saddle.
If a small displacement returns to the same minimum on both sides, the script
tries progressively larger displacements automatically.
"""

from pathlib import Path
import numpy as np
from ase.io import read, write
from ase.optimize import FIRE

from common import (
    make_calculator, load_json, dump_json, normalized, rmsd_mic,
    infer_vacancy_sites, vacancy_separation
)

ROOT = Path(__file__).resolve().parent
RES = ROOT / 'results'
DOUT = RES / 'dimer'
MDIR = RES / 'mechanisms'
MDIR.mkdir(parents=True, exist_ok=True)

selection = load_json(RES / 'selected_saddles.json')['selected']
perfect = read(RES / 'perfect_fcc.traj')
start = read(RES / 'divacancy_relaxed.traj')


def relax_structure(atoms, log, traj):
    atoms = atoms.copy()
    atoms.calc = make_calculator()
    opt = FIRE(atoms, logfile=str(log), trajectory=str(traj))
    opt.run(fmax=0.02, steps=1200)
    e = float(atoms.get_potential_energy())
    return atoms, e

all_results = []
for item in selection:
    search = item['search']
    name = item['name']
    out = MDIR / name
    out.mkdir(exist_ok=True)

    saddle = read(DOUT / f'{search}_saddle.traj')
    saddle.calc = make_calculator()
    saddle_energy = float(saddle.get_potential_energy())
    mode = normalized(np.load(DOUT / f'{search}_eigenmode.npy'))

    accepted = None
    for delta in [0.12, 0.20, 0.30, 0.45, 0.65]:
        plus = saddle.copy()
        minus = saddle.copy()
        plus.set_positions(plus.positions + delta * mode)
        minus.set_positions(minus.positions - delta * mode)

        plus, eplus = relax_structure(plus, out / f'plus_{delta:.2f}.log', out / f'plus_{delta:.2f}.traj')
        minus, eminus = relax_structure(minus, out / f'minus_{delta:.2f}.log', out / f'minus_{delta:.2f}.traj')
        endpoint_rmsd = rmsd_mic(plus, minus)
        if endpoint_rmsd > 0.05:
            accepted = (delta, plus, eplus, minus, eminus, endpoint_rmsd)
            break

    if accepted is None:
        raise RuntimeError(
            f'{name}/{search}: both sides of the saddle relaxed to effectively the same minimum. '
            'Choose a different Dimer saddle in 03_analyze_select_saddles_template.py.'
        )

    delta, plus, eplus, minus, eminus, endpoint_rmsd = accepted

    # Label state A as whichever endpoint is closer to the original relaxed divacancy.
    rplus = rmsd_mic(plus, start)
    rminus = rmsd_mic(minus, start)
    if rplus <= rminus:
        A, EA, B, EB = plus, eplus, minus, eminus
    else:
        A, EA, B, EB = minus, eminus, plus, eplus

    write(out / 'state_A.traj', A)
    write(out / 'state_B.traj', B)
    write(out / 'dimer_saddle.traj', saddle)

    miss_A = infer_vacancy_sites(perfect, A)
    miss_B = infer_vacancy_sites(perfect, B)
    sep_A = vacancy_separation(perfect, miss_A)
    sep_B = vacancy_separation(perfect, miss_B)

    result = {
        'name': name,
        'search': search,
        'downhill_displacement_A': float(delta),
        'endpoint_rmsd_A': float(endpoint_rmsd),
        'state_A_energy_eV': float(EA),
        'state_B_energy_eV': float(EB),
        'saddle_energy_eV': float(saddle_energy),
        'dimer_forward_barrier_eV': float(saddle_energy - EA),
        'dimer_reverse_barrier_eV': float(saddle_energy - EB),
        'state_A_vacancy_sites': [int(x) for x in miss_A],
        'state_B_vacancy_sites': [int(x) for x in miss_B],
        'state_A_vacancy_separation_A': float(sep_A),
        'state_B_vacancy_separation_A': float(sep_B),
    }
    dump_json(out / 'endpoints.json', result)
    all_results.append(result)

    print(f'\n{name} ({search})')
    print(f'  downhill displacement: {delta:.2f} A')
    print(f'  state A vacancy separation: {sep_A:.4f} A')
    print(f'  state B vacancy separation: {sep_B:.4f} A')
    print(f'  Dimer forward barrier: {saddle_energy - EA:.6f} eV')

# Index of discovered transitions.
dump_json(MDIR / 'mechanism_index.json', {'mechanisms': all_results})
print('\nDownhill relaxations complete. The two Dimer saddles now have known endpoint minima.')
