#!/usr/bin/env python3
"""
Question 4B: Run an ASE nudged elastic band calculation for vacancy migration.

This script is supplied COMPLETE. Students should not need to modify it.

Before running:
  * Complete 01_build_vacancy_endpoints_template.py.
  * The instructor must provide an EAM/alloy potential named Al.eam.alloy
    in this directory (or edit POTENTIAL_FILE below before distribution).

Outputs are written to results/.
"""

from pathlib import Path
import csv
import numpy as np

from ase.calculators.eam import EAM
from ase.geometry import find_mic
from ase.io import read, write
from ase.mep import NEB
from ase.optimize import FIRE

ROOT = Path(__file__).resolve().parent
ENDPOINT_DIR = ROOT / "endpoints"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

POTENTIAL_FILE = ROOT / "Al_zhou.eam.alloy"
N_IMAGES = 7                    # includes initial and final images
ENDPOINT_FMAX = 0.03            # eV/A
NEB_FMAX = 0.03                 # eV/A
MAX_ENDPOINT_STEPS = 1000
MAX_NEB_STEPS = 2000


def calculator():
    """Create a fresh ASE EAM calculator for one structure/image."""
    return EAM(potential=str(POTENTIAL_FILE))


def relax_endpoint(atoms, name):
    """Relax atomic positions at fixed cell and return the relaxed structure."""
    atoms = atoms.copy()
    atoms.calc = calculator()
    opt = FIRE(
        atoms,
        trajectory=str(RESULTS_DIR / f"{name}_relax.traj"),
        logfile=str(RESULTS_DIR / f"{name}_relax.log"),
    )
    opt.run(fmax=ENDPOINT_FMAX, steps=MAX_ENDPOINT_STEPS)
    write(RESULTS_DIR / f"{name}_relaxed.traj", atoms)
    return atoms


def reaction_coordinate(images):
    """Cumulative geometric path length, normalized from 0 to 1."""
    s = [0.0]
    for left, right in zip(images[:-1], images[1:]):
        dr = right.positions - left.positions
        mic_dr, _ = find_mic(dr, cell=left.cell, pbc=left.pbc)
        ds = float(np.sqrt(np.sum(mic_dr ** 2)))
        s.append(s[-1] + ds)
    s = np.asarray(s)
    if s[-1] <= 0.0:
        raise RuntimeError("NEB path has zero total length.")
    return s / s[-1]


if not POTENTIAL_FILE.exists():
    raise FileNotFoundError(
        f"Could not find {POTENTIAL_FILE.name}. The instructor must place the "
        "Al EAM/alloy potential in the problem directory."
    )

initial_file = ENDPOINT_DIR / "initial.traj"
final_file = ENDPOINT_DIR / "final.traj"

if not initial_file.exists() or not final_file.exists():
    raise FileNotFoundError(
        "Missing endpoint files. Complete and run "
        "01_build_vacancy_endpoints_template.py first."
    )

initial = read(initial_file)
final = read(final_file)

if len(initial) != len(final):
    raise RuntimeError("NEB endpoints must contain the same number of atoms.")

print("Relaxing initial endpoint...")
initial = relax_endpoint(initial, "initial")

print("Relaxing final endpoint...")
final = relax_endpoint(final, "final")

initial_energy = initial.get_potential_energy()
final_energy = final.get_potential_energy()
print(f"Relaxed endpoint energy difference E_final-E_initial = "
      f"{final_energy - initial_energy:+.6e} eV")

# Build the band. Copies are essential: each image must be a distinct Atoms object.
images = [initial.copy()]
images += [initial.copy() for _ in range(N_IMAGES - 2)]
images += [final.copy()]

# Every NEB image gets its own calculator instance.
for image in images:
    image.calc = calculator()

neb = NEB(images, climb=True, method="improvedtangent")
neb.interpolate(mic=True)

print(f"Running climbing-image NEB with {N_IMAGES} images...")
opt = FIRE(
    neb,
    trajectory=str(RESULTS_DIR / "vacancy_neb.traj"),
    logfile=str(RESULTS_DIR / "vacancy_neb.log"),
)
opt.run(fmax=NEB_FMAX, steps=MAX_NEB_STEPS)

# Save the final converged band as a simple list of images.
write(RESULTS_DIR / "neb_final_images.traj", images)

energies = np.asarray([image.get_potential_energy() for image in images])
coordinate = reaction_coordinate(images)

with open(RESULTS_DIR / "neb_energies.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "reaction_coordinate", "energy_eV"])
    for i, (s, e) in enumerate(zip(coordinate, energies)):
        writer.writerow([i, f"{s:.10f}", f"{e:.12f}"])

print("\nNEB calculation finished")
print("========================")
print(f"Initial energy: {energies[0]:.8f} eV")
print(f"Final energy:   {energies[-1]:.8f} eV")
print("\nCreated:")
print("  results/neb_energies.csv")
print("  results/neb_final_images.traj")
print("  results/vacancy_neb.traj")
print("  results/vacancy_neb.log")
print("\nNext: complete 03_analyze_neb_template.py")
