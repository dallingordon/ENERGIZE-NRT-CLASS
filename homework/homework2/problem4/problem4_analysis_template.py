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
    history = []
    with open(path) as f:
        lines = f.readlines()

    energies, max_forces = [], []
    for i, line in enumerate(lines):
        # One "free  energy   TOTEN" line (two spaces) is printed per IONIC step.
        # (The per-electronic-step lines read "free energy    TOTEN", so they don't match.)
        if "free  energy   TOTEN" in line:
            energies.append(float(line.split()[-2]))

        # Force block: header, a dashed line, then one row per atom:
        #   x y z  Fx Fy Fz   -- ends at the next dashed line
        if "TOTAL-FORCE (eV/Angst)" in line:
            j = i + 2
            forces = []
            while not lines[j].strip().startswith("---"):
                forces.append([float(v) for v in lines[j].split()[3:6]])
                j += 1
            forces = np.array(forces)
            # |F| per atom, keep the largest one for this ionic step
            max_forces.append(np.linalg.norm(forces, axis=1).max())

    for k, (e, fmax) in enumerate(zip(energies, max_forces), start=1):
        history.append({"ionic_step": k, "energy_eV": e, "max_force_eV_A": fmax})
    return history


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
    counts = []
    n = 0
    with open(path) as f:
        for line in f:
            tok = line.split()
            if not tok:
                continue
            if tok[0] in ("DAV:", "RMM:", "CG:"):
                n += 1                      # one electronic (SCF) iteration
            elif len(tok) > 1 and tok[1] == "F=":
                counts.append(n)            # ionic step finished -> store and reset
                n = 0
    return counts


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
    # 1. fractional displacement of each atom
    dfrac = frac_final - frac_initial
    # 2. minimum image: the cell repeats, so a step of 0.985 is really -0.015.
    #    Subtracting the nearest integer wraps every component into [-0.5, 0.5].
    dfrac -= np.rint(dfrac)
    # 3. fractional -> Cartesian (Angstrom)
    dcart = dfrac @ cell
    # 4. length of each displacement vector
    dmag = np.linalg.norm(dcart, axis=1)
    return dcart, dmag


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
    # The cell is only 5.43 A wide, so plain minimum-image distances stop at
    # ~2.7 A (half the cell). To see the 2nd and 3rd neighbour shells we instead
    # place copies of the cell around the original (a 5x5x5 block of images)
    # and measure distances from each atom in the central cell to every atom
    # in every copy.
    r_max = 6.0
    n = len(frac_coords)
    cart = frac_coords @ cell
    shifts = np.array([[i, j, k] for i in range(-2, 3)
                                 for j in range(-2, 3)
                                 for k in range(-2, 3)]) @ cell

    dists = []
    for a in range(n):
        for b in range(n):
            # vector from atom a to every periodic image of atom b
            d = np.linalg.norm(cart[b] + shifts - cart[a], axis=1)
            dists.extend(d[(d > 1e-8) & (d < r_max)])   # skip self-distance 0
    dists = np.array(dists)

    edges = np.arange(0.0, r_max + dr, dr)
    hist, _ = np.histogram(dists, bins=edges)
    r = 0.5 * (edges[1:] + edges[:-1])

    # Normalize: divide by what an ideal gas of the same density would give,
    # so g(r) -> 1 at large r and peaks show neighbour shells.
    volume = abs(np.linalg.det(cell))
    rho = n / volume
    shell_vol = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    g = hist / (n * rho * shell_vol)
    return r, g


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
    # 2. Read the initial (POSCAR) and relaxed (CONTCAR) structures.
    #    CONTCAR is VASP's output, written in the same format as POSCAR.
    # --------------------------------------------------------
    cell0, frac0, cart0, labels0 = read_poscar(ROOT / "POSCAR")
    cell1, frac1, cart1, labels1 = read_poscar(ROOT / "CONTCAR")

    # --------------------------------------------------------
    # 3. Compute atomic displacements
    # --------------------------------------------------------
    dcart, dmag = minimum_image_displacements(frac0, frac1, cell0)

    largest_atom_index = np.argmax(dmag)
    largest_displacement = dmag[largest_atom_index]

    print("Per-atom displacement (A):")
    for i, (lab, d) in enumerate(zip(labels0, dmag), start=1):
        print(f"  atom {i} ({lab}): {d:.4f}")

    # --------------------------------------------------------
    # 4. Extract ionic-step energy / force history
    # --------------------------------------------------------
    history = parse_outcar_ionic_history(ROOT / "OUTCAR")

    steps = np.array([row["ionic_step"] for row in history])
    energies = np.array([row["energy_eV"] for row in history], dtype=float)
    forces = np.array([row["max_force_eV_A"] for row in history], dtype=float)

    # --------------------------------------------------------
    # 5. Extract SCF iteration counts from OSZICAR
    # --------------------------------------------------------
    if (ROOT / "OSZICAR").exists():
        scf_counts = parse_oszicar_scf_iterations(ROOT / "OSZICAR")
    else:
        scf_counts = []

    # --------------------------------------------------------
    # 6. Plot energy and force vs ionic step
    # --------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(steps, energies, marker="o")
    axes[0].set_xlabel("Ionic step")
    axes[0].set_ylabel("Total energy (eV)")
    axes[0].set_title("Total energy vs ionic step")

    axes[1].plot(steps, forces, marker="o")
    axes[1].axhline(0.02, ls="--", c="gray", label="EDIFFG = 0.02 eV/Å")
    axes[1].set_xlabel("Ionic step")
    axes[1].set_ylabel("Maximum force (eV/Å)")
    axes[1].set_title("Maximum force vs ionic step")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(ROOT / "problem4_geometry_optimization.png", dpi=200)

    # --------------------------------------------------------
    # 7. Compute and plot RDFs for initial and final structures
    # --------------------------------------------------------
    r0, g0 = compute_rdf(frac0, cell0)
    r1, g1 = compute_rdf(frac1, cell1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(r0, g0, label="Initial POSCAR")
    ax.plot(r1, g1, label="Final CONTCAR")
    ax.set_xlabel(r"$r$ ($\AA$)")
    ax.set_ylabel(r"$g(r)$")
    ax.set_title("Initial vs final radial distribution function")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "problem4_RDF.png", dpi=200)

    # --------------------------------------------------------
    # 8. Save a small summary table
    # --------------------------------------------------------
    summary_rows = [
        ("ionic_steps", len(history)),
        ("initial_energy_eV", energies[0]),
        ("final_energy_eV", energies[-1]),
        ("final_max_force_eV_A", forces[-1]),
        ("largest_displacement_A", largest_displacement),
        ("largest_displacement_atom", largest_atom_index + 1),
    ]

    if scf_counts:
        summary_rows.append(("first_ionic_step_scf_iterations", scf_counts[0]))
        summary_rows.append(("last_ionic_step_scf_iterations", scf_counts[-1]))

    save_summary_table(ROOT / "problem4_summary.csv", summary_rows)

    # --------------------------------------------------------
    # 9. Print a concise console summary
    # --------------------------------------------------------
    print(f"Ionic steps: {len(history)}")
    print(f"Initial energy: {energies[0]:.6f} eV")
    print(f"Final energy:   {energies[-1]:.6f} eV")
    print(f"Final max force:{forces[-1]:.6f} eV/A")
    print(f"Largest displacement: {largest_displacement:.6f} A (atom {largest_atom_index + 1})")
    print(f"SCF iterations per ionic step: {scf_counts}")


if __name__ == "__main__":
    main()
