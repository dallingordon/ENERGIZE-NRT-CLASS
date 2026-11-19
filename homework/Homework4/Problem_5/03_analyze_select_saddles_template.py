#!/usr/bin/env python3
"""03 - Compare the Dimer searches and select two distinct saddles.

The supplied code makes a barrier plot and a pairwise structural RMSD table.

STUDENT TASK:
  Inspect the saddle trajectories (ASE GUI or OVITO), use the barrier/RMSD
  information, and select TWO physically distinct saddle points for the
  downhill-relaxation + NEB portion of the problem.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ase.io import read

from common import rmsd_mic, dump_json

ROOT = Path(__file__).resolve().parent
DOUT = ROOT / 'results' / 'dimer'
summary = pd.read_csv(DOUT / 'dimer_search_summary.csv')

print('\nDimer search summary:')
print(summary.to_string(index=False))

# Barrier plot.
plt.figure(figsize=(7, 4.5))
plt.scatter(summary['label'], summary['barrier_from_start_eV'])
plt.xlabel('Dimer search')
plt.ylabel('Barrier from starting divacancy (eV)')
plt.title('Nearby saddle points found from the same Al divacancy')
plt.tight_layout()
plt.savefig(DOUT / 'dimer_barriers.png', dpi=200)
plt.close()

# Pairwise saddle RMSD, useful for spotting searches that converged to the same structure.
labels = summary['label'].tolist()
saddles = {lab: read(DOUT / f'{lab}_saddle.traj') for lab in labels}
mat = np.zeros((len(labels), len(labels)))
for i, a in enumerate(labels):
    for j, b in enumerate(labels):
        mat[i, j] = rmsd_mic(saddles[a], saddles[b])

rmsd_df = pd.DataFrame(mat, index=labels, columns=labels)
rmsd_df.to_csv(DOUT / 'saddle_rmsd_matrix_A.csv')
print('\nPairwise saddle RMSD (A):')
print(rmsd_df.round(4).to_string())

# =====================================================================
# TODO - SELECT TWO DISTINCT DIMER SADDLES
# =====================================================================
# First inspect:
#   results/dimer/dimer_barriers.png
#   results/dimer/saddle_rmsd_matrix_A.csv
#   results/dimer/S##_saddle.traj
#
# Choose two searches that correspond to physically distinct saddle points.
# Give each one a short mechanism name of your choosing.
#
# Example FORMAT ONLY:
# SELECTED = [
#     {'search': 'S01', 'name': 'mechanism_A'},
#     {'search': 'S04', 'name': 'mechanism_B'},
# ]
#
SELECTED = []

if len(SELECTED) != 2:
    raise NotImplementedError('Select exactly two distinct Dimer saddles in SELECTED.')

available = set(labels)
for item in SELECTED:
    if item['search'] not in available:
        raise ValueError(f"Unknown Dimer search {item['search']}")

# Save the selection for all subsequent scripts.
dump_json(ROOT / 'results' / 'selected_saddles.json', {'selected': SELECTED})
print('\nSelected saddles:')
for item in SELECTED:
    print(f"  {item['name']}: {item['search']}")
print('Wrote results/selected_saddles.json')
