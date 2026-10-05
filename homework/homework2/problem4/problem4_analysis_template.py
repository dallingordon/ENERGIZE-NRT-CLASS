#!/usr/bin/env python3
"""
Homework 2 - Problem 4
STUDENT TEMPLATE

Goal:
    Analyze a VASP geometry optimization of distorted Si.

Expected files in the same directory:
    POSCAR
    CONTCAR
    OUTCAR
    OSZICAR   (recommended)

What you need to produce for the assignment:
    1. problem4_geometry_optimization.png
       - total energy vs ionic step
       - maximum atomic force vs ionic step

    2. RDF plots for the initial and final structures
       - you can either save these separately or combine them in one figure

    3. A small table containing:
       - number of ionic steps
       - initial energy
       - final energy
       - final maximum force
       - largest atomic displacement

    4. Short written interpretation (done separately by you, not in this script)

This file is a TEMPLATE.
Many sections are intentionally incomplete and marked TODO.
"""

from pathlib import Path
import re
import csv
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


# ============================================================
# FILE READING HELPERS
# ============================================================


def read_poscar(path):
    """
    Read a POSCAR/CONTCAR file.

    TODO:
        Implement a basic POSCAR reader.

    Suggested outputs:
        cell   : (3,3) numpy array
        frac   : (N,3) fractional coordinates
        cart   : (N,3) Cartesian coordinates
        labels : list of atomic symbols for each atom

    Hints:
        - line 1: title
        - line 2: scaling factor
        - lines 3-5: lattice vectors
        - next line(s): species and counts
        - then coordinate mode: Direct or Cartesian
        - then atomic coordinates
    """
    with open(path) as f:
        lines = [line.split() for line in f]

    # line 1: title
    title = " ".join(lines[0])

    # line 2: scaling factor
    scale = float(lines[1][0])

    # lines 3-5: lattice vectors
    cell = np.array([[float(x) for x in lines[i][:3]] for i in range(2, 5)]) * scale

    # lines 6-7: species and counts
    species = lines[5]
    counts = [int(c) for c in lines[6]]
    n_atoms = sum(counts)
    labels = [sp for sp, n in zip(species, counts) for _ in range(n)]

    # Selective
    idx = 7
    selective_dynamics = lines[idx][0][0].lower() == "s"
    if selective_dynamics:
        idx += 1

    # coordinate mode: Direct or Cartesian
    mode = lines[idx][0][0].lower()
    is_direct = mode == "d"
    idx += 1

    # atomic coordinates 3d coord
    coords = np.array([[float(x) for x in lines[idx + i][:3]] for i in range(n_atoms)])

    if is_direct:
        frac = coords
        cart = frac @ cell
    else:
        cart = coords * scale
        frac = cart @ np.linalg.inv(cell)

    ##raise NotImplementedError("TODO: implement read_poscar()")
    return cell, frac, cart, labels

def parse_outcar_ionic_history(path):
    """
    Extract one row per ionic step from OUTCAR.

    TODO:
        Parse:
            - total energy at each ionic step
            - maximum force magnitude at each ionic step

    Suggested return:
        list of dicts, e.g.
        [
            {"ionic_step": 1, "energy_eV": ..., "max_force_eV_A": ...},
            {"ionic_step": 2, "energy_eV": ..., "max_force_eV_A": ...},
            ...
        ]

    Hints:
        - Search for lines containing 'TOTEN'
        - Search for blocks beginning with 'TOTAL-FORCE (eV/Angst)'
        - In each force block, compute the force magnitude of every atom:
              |F| = sqrt(Fx^2 + Fy^2 + Fz^2)
          and keep the maximum for that ionic step.
    """
    raise NotImplementedError("TODO: implement parse_outcar_ionic_history()")


def parse_oszicar_scf_iterations(path):
    """
    OPTIONAL / RECOMMENDED:
    Count how many electronic SCF iterations occurred during each ionic step.

    TODO:
        Parse OSZICAR and count lines starting with things like:
            DAV:
            RMM:
            CG:

        These lines occur before a line with something like:
            1 F= ...
            2 F= ...
        which indicates the ionic step summary.

    Suggested return:
        list of integers, one per ionic step
    """
    raise NotImplementedError("TODO: implement parse_oszicar_scf_iterations()")


# ============================================================
# GEOMETRY / STRUCTURE ANALYSIS
# ============================================================

def minimum_image_displacements(frac_initial, frac_final, cell):
    """
    Compute per-atom displacement vectors and magnitudes using
    periodic boundary conditions.

    TODO:
        1. Compute fractional displacement:
               dfrac = frac_final - frac_initial
        2. Wrap into minimum image using:
               dfrac -= np.rint(dfrac)
        3. Convert to Cartesian:
               dcart = dfrac @ cell
        4. Compute magnitudes with np.linalg.norm(..., axis=1)

    Return:
        dcart, dmag
    """
    raise NotImplementedError("TODO: implement minimum_image_displacements()")


def compute_rdf(frac_coords, cell, dr=0.03):
    """
    Compute a simple radial distribution function g(r) for one structure.

    TODO:
        1. Loop over all unique atom pairs
        2. Compute pair distances with minimum-image convention
        3. Histogram the distances
        4. Normalize to obtain g(r)

    Return:
        r, g

    Hints:
        - A simple structure-level RDF is sufficient for this homework.
        - You are comparing the initial and final structures, so consistency
          matters more than extreme sophistication.
    """
    raise NotImplementedError("TODO: implement compute_rdf()")


# ============================================================
# WRITING OUTPUT TABLES
# ============================================================

def save_summary_table(filename, summary_rows):
    """
    Save a small summary table as CSV.

    Parameters:
        filename     : output file name
        summary_rows : list of (quantity, value) pairs
    """
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["quantity", "value"])
        writer.writerows(summary_rows)


# ============================================================
# MAIN WORKFLOW
# ============================================================

def main():
    # --------------------------------------------------------
    # 1. Check input files
    # --------------------------------------------------------
    required = ["POSCAR", "CONTCAR", "OUTCAR"]
    missing = [name for name in required if not (ROOT / name).exists()]
    if missing:
        raise SystemExit(f"Missing required files: {', '.join(missing)}")

    # --------------------------------------------------------
    # 2. Read the initial and final structures
    # --------------------------------------------------------
    # TODO: uncomment after implementing read_poscar()
    cell0, frac0, cart0, labels0 = read_poscar(ROOT / "POSCAR")
    ##cell1, frac1, cart1, labels1 = read_poscar(ROOT / "CONTCAR") #this is an scc output?
    breakpoint()

    # --------------------------------------------------------
    # 3. Compute atomic displacements
    # --------------------------------------------------------
    # TODO: uncomment after implementing minimum_image_displacements()
    # dcart, dmag = minimum_image_displacements(frac0, frac1, cell0)
    #
    # largest_atom_index = np.argmax(dmag)
    # largest_displacement = dmag[largest_atom_index]

    # --------------------------------------------------------
    # 4. Extract ionic-step energy / force history
    # --------------------------------------------------------
    # TODO: uncomment after implementing parse_outcar_ionic_history()
    # history = parse_outcar_ionic_history(ROOT / "OUTCAR")
    #
    # steps = np.array([row["ionic_step"] for row in history])
    # energies = np.array([row["energy_eV"] for row in history], dtype=float)
    # forces = np.array([row["max_force_eV_A"] for row in history], dtype=float)

    # --------------------------------------------------------
    # 5. OPTIONAL: Extract SCF iteration counts from OSZICAR
    # --------------------------------------------------------
    # if (ROOT / "OSZICAR").exists():
    #     scf_counts = parse_oszicar_scf_iterations(ROOT / "OSZICAR")
    # else:
    #     scf_counts = []

    # --------------------------------------------------------
    # 6. Plot energy and force vs ionic step
    # --------------------------------------------------------
    # TODO: make the required figure and save it as:
    #       problem4_geometry_optimization.png
    #
    # Suggested structure:
    #
    # fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    #
    # axes[0].plot(steps, energies, marker="o")
    # axes[0].set_xlabel("Ionic step")
    # axes[0].set_ylabel("Total energy (eV)")
    # axes[0].set_title("Total energy vs ionic step")
    #
    # axes[1].plot(steps, forces, marker="o")
    # axes[1].set_xlabel("Ionic step")
    # axes[1].set_ylabel("Maximum force (eV/Å)")
    # axes[1].set_title("Maximum force vs ionic step")
    #
    # fig.tight_layout()
    # fig.savefig(ROOT / "problem4_geometry_optimization.png", dpi=200)

    # --------------------------------------------------------
    # 7. Compute and plot RDFs for initial and final structures
    # --------------------------------------------------------
    # TODO:
    # r0, g0 = compute_rdf(frac0, cell0)
    # r1, g1 = compute_rdf(frac1, cell1)
    #
    # fig, ax = plt.subplots(figsize=(7, 4.5))
    # ax.plot(r0, g0, label="Initial POSCAR")
    # ax.plot(r1, g1, label="Final CONTCAR")
    # ax.set_xlabel(r"$r$ ($\AA$)")
    # ax.set_ylabel(r"$g(r)$")
    # ax.set_title("Initial vs final radial distribution function")
    # ax.legend()
    # fig.tight_layout()
    # fig.savefig(ROOT / "problem4_RDF.png", dpi=200)

    # --------------------------------------------------------
    # 8. Save a small summary table
    # --------------------------------------------------------
    # TODO:
    # summary_rows = [
    #     ("ionic_steps", len(history)),
    #     ("initial_energy_eV", energies[0]),
    #     ("final_energy_eV", energies[-1]),
    #     ("final_max_force_eV_A", forces[-1]),
    #     ("largest_displacement_A", largest_displacement),
    #     ("largest_displacement_atom", largest_atom_index + 1),
    # ]
    #
    # if scf_counts:
    #     summary_rows.append(("first_ionic_step_scf_iterations", scf_counts[0]))
    #     summary_rows.append(("last_ionic_step_scf_iterations", scf_counts[-1]))
    #
    # save_summary_table(ROOT / "problem4_summary.csv", summary_rows)

    # --------------------------------------------------------
    # 9. Print a concise console summary
    # --------------------------------------------------------
    # TODO:
    # print(f"Ionic steps: {len(history)}")
    # print(f"Initial energy: {energies[0]:.6f} eV")
    # print(f"Final energy:   {energies[-1]:.6f} eV")
    # print(f"Final max force:{forces[-1]:.6f} eV/A")
    # print(f"Largest displacement: {largest_displacement:.6f} A")
    #
    print("Problem 4 template loaded.")
    print("Complete the TODO sections to perform the full analysis.")


if __name__ == "__main__":
    main()
