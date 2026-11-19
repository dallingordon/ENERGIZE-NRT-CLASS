#!/usr/bin/env python3
"""05 - Run NEB between the two minima discovered from each Dimer saddle.

This script is complete. Each calculation uses 7 total images, first relaxes a
regular NEB, then refines with climbing-image NEB.
"""

from pathlib import Path
from ase.io import read, write
from ase.mep import NEB
from ase.optimize import FIRE

from common import make_calculator, load_json, dump_json

ROOT = Path(__file__).resolve().parent
RES = ROOT / 'results'
MDIR = RES / 'mechanisms'
index = load_json(MDIR / 'mechanism_index.json')['mechanisms']

N_IMAGES = 7

for entry in index:
    name = entry['name']
    out = MDIR / name
    A = read(out / 'state_A.traj')
    B = read(out / 'state_B.traj')

    # Attach calculators to endpoints so their energies are available.
    A.calc = make_calculator()
    B.calc = make_calculator()
    EA = float(A.get_potential_energy())
    EB = float(B.get_potential_energy())

    images = [A]
    images += [A.copy() for _ in range(N_IMAGES - 2)]
    images += [B]
    for image in images[1:-1]:
        image.calc = make_calculator()

    neb = NEB(images, climb=False, method='improvedtangent')
    neb.interpolate(method='idpp', mic=True, apply_constraint=True)
    pre = FIRE(neb, logfile=str(out / 'neb_pre.log'), trajectory=str(out / 'neb_pre.traj'))
    pre.run(fmax=0.08, steps=300)

    # Re-wrap the same images in climbing-image NEB for saddle refinement.
    neb_ci = NEB(images, climb=True, method='improvedtangent')
    opt = FIRE(neb_ci, logfile=str(out / 'neb_ci.log'), trajectory=str(out / 'neb_ci.traj'))
    converged = bool(opt.run(fmax=0.04, steps=700))

    energies = [float(img.get_potential_energy()) for img in images]
    write(out / 'neb_final_images.traj', images)

    result = {
        'name': name,
        'converged': converged,
        'n_images': N_IMAGES,
        'image_energies_eV': energies,
        'state_A_energy_eV': EA,
        'state_B_energy_eV': EB,
        'max_image_energy_eV': max(energies),
        'neb_forward_barrier_eV': max(energies) - EA,
        'neb_reverse_barrier_eV': max(energies) - EB,
    }
    dump_json(out / 'neb_results.json', result)
    print(f"{name}: NEB converged={converged}, forward barrier={result['neb_forward_barrier_eV']:.6f} eV")

print('\nNEB calculations complete.')
