#!/usr/bin/env python3
"""06 - Compare Dimer saddle energies with the NEB pathways.

STUDENT TASKS:
  1. Calculate the Dimer and NEB forward barriers for both mechanisms.
  2. Calculate the absolute barrier difference.
  3. Plot both NEB energy profiles and mark the corresponding Dimer saddle energy.

The script supplies all file reading and final CSV output structure.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import load_json

ROOT = Path(__file__).resolve().parent
MDIR = ROOT / 'results' / 'mechanisms'
index = load_json(MDIR / 'mechanism_index.json')['mechanisms']

rows = []
fig, axes = plt.subplots(1, len(index), figsize=(6.2 * len(index), 4.6), squeeze=False)

for col, mechanism in enumerate(index):
    name = mechanism['name']
    out = MDIR / name
    endpoints = load_json(out / 'endpoints.json')
    neb = load_json(out / 'neb_results.json')

    EA = endpoints['state_A_energy_eV']
    E_saddle = endpoints['saddle_energy_eV']
    neb_energies = np.array(neb['image_energies_eV'], float)

    # =================================================================
    # TODO 1 - CALCULATE THE TWO FORWARD BARRIERS
    # =================================================================
    # Dimer: saddle energy - state A energy
    # NEB:   highest NEB image energy - state A energy
    #
    dimer_barrier = E_saddle - EA
    neb_barrier = np.max(neb_energies) - EA

    if dimer_barrier is None or neb_barrier is None:
        raise NotImplementedError('TODO 1: calculate Dimer and NEB forward barriers.')

    # =================================================================
    # TODO 2 - CALCULATE THEIR ABSOLUTE DIFFERENCE
    # =================================================================
    barrier_difference = abs(dimer_barrier - neb_barrier)
    if barrier_difference is None:
        raise NotImplementedError('TODO 2: calculate the absolute barrier difference.')

    reaction_coordinate = np.linspace(0.0, 1.0, len(neb_energies))
    relative_neb = neb_energies - EA
    relative_dimer = E_saddle - EA

    # =================================================================
    # TODO 3 - PLOT NEB PROFILE + DIMER SADDLE
    # =================================================================
    # On axes[0, col]:
    #   * plot reaction_coordinate vs relative_neb with markers
    #   * draw a horizontal line at relative_dimer
    #   * label axes and add a legend/title
    #
    ax = axes[0, col]
    ax.plot(reaction_coordinate, relative_neb, 'o-', label=f'NEB (Ea = {neb_barrier:.3f} eV)')
    ax.axhline(relative_dimer, ls='--', color='tab:red', label=f'Dimer saddle (Ea = {dimer_barrier:.3f} eV)')
    ax.set_xlabel('Normalized reaction coordinate')
    ax.set_ylabel('Energy relative to state A (eV)')
    ax.set_title(f"{name} ({mechanism['search']})")
    ax.legend()

    rows.append({
        'mechanism': name,
        'dimer_barrier_eV': float(dimer_barrier),
        'neb_barrier_eV': float(neb_barrier),
        'absolute_difference_eV': float(barrier_difference),
        'state_A_vacancy_separation_A': endpoints['state_A_vacancy_separation_A'],
        'state_B_vacancy_separation_A': endpoints['state_B_vacancy_separation_A'],
    })

plt.tight_layout()
plt.savefig(MDIR / 'dimer_vs_neb.png', dpi=200)
plt.close()

comparison = pd.DataFrame(rows)
comparison.to_csv(MDIR / 'dimer_vs_neb_comparison.csv', index=False)

print('\nDimer vs NEB comparison:')
print(comparison.to_string(index=False))
print('\nCreated:')
print('  results/mechanisms/dimer_vs_neb.png')
print('  results/mechanisms/dimer_vs_neb_comparison.csv')
