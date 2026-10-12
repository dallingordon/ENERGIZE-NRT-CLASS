#!/usr/bin/env python3
"""
Homework 2 - Problem 5
STUDENT TEMPLATE

Goal:
    Compare the electronic density of states (DOS) of Al and Si.

Suggested directory layout:
    problem5/
      problem5_analysis_template.py
      aluminum/
        dos/
          DOSCAR
          OUTCAR
      silicon/
        dos/
          DOSCAR
          OUTCAR

What you need to produce for the assignment:
    1. problem5_Al_vs_Si_DOS.png
    2. A comparison table with at least:
       - material
       - Fermi energy
       - DOS at EF
       - calculated gap
       - electronic classification
    3. Short written interpretation (done separately by you, not in this script)

This file is a TEMPLATE.
Many sections are intentionally incomplete and marked TODO.
"""

from pathlib import Path
import argparse
import csv
import re
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# FILE READING HELPERS
# ============================================================

def parse_fermi_from_outcar(path):
    """
    Extract the Fermi energy from OUTCAR.

    TODO:
        Search for lines containing:
            E-fermi : 5.4321 ...
        and extract the numerical value.

    Return:
        float efermi
    """
    efermi = None
    with open(path) as f:
        for line in f:
            if "E-fermi" in line:
                # e.g. " E-fermi :   7.9491     XC(G=0): ..."
                efermi = float(line.split()[2])   # keep the LAST occurrence
    if efermi is None:
        raise ValueError(f"No E-fermi line found in {path}")
    return efermi


def read_doscar(path):
    """
    Read the total DOS from DOSCAR.

    TODO:
        Parse:
            - energy values
            - total DOS values

    Suggested return:
        energy, dos, efermi_from_header

    Hints:
        - DOSCAR begins with several header lines
        - One line contains NEDOS and Efermi
        - The following lines contain the DOS data
        - For a simple non-spin-polarized case, the columns are usually:
              E   DOS   integrated_DOS
        - If you want to be more robust, also handle spin-polarized data
    """
    with open(path) as f:
        lines = f.readlines()

    # Line 1: first number is the number of atoms in the cell
    n_atoms = int(lines[0].split()[0])

    # Line 6 (index 5): Emax  Emin  NEDOS  Efermi  weight
    header = lines[5].split()
    nedos = int(header[2])
    efermi_from_header = float(header[3])

    # The next NEDOS lines are the TOTAL DOS block.
    # (With LORBIT=11, per-atom projected blocks follow; we ignore those.)
    data = np.array([[float(x) for x in lines[6 + i].split()] for i in range(nedos)])
    energy = data[:, 0]
    if data.shape[1] == 3:          # non-spin-polarized: E  DOS  intDOS
        dos = data[:, 1]
    else:                           # spin-polarized: E  DOSup  DOSdown  intUp  intDown
        dos = data[:, 1] + data[:, 2]

    return energy, dos, efermi_from_header, n_atoms


# ============================================================
# DOS ANALYSIS HELPERS
# ============================================================

def shift_energy_axis(energy, efermi):
    """
    Shift energies so that EF = 0 eV.

    TODO:
        return energy - efermi
    """
    return energy - efermi


def dos_at_fermi(energy_shifted, dos):
    """
    Estimate the DOS at EF = 0.

    TODO:
        Use interpolation to estimate DOS at 0 eV.

    Hint:
        np.interp(0.0, energy_shifted, dos)
    """
    # DOS is on a discrete energy grid; linearly interpolate to exactly 0 eV
    return float(np.interp(0.0, energy_shifted, dos))


def estimate_gap_from_dos(energy_shifted, dos, threshold_fraction=0.01):
    """
    Estimate the Si band gap from the DOS.

    TODO:
        Use a small threshold relative to the maximum DOS.
        Then:
            - find the last occupied energy below 0 eV with DOS above threshold
            - find the first unoccupied energy above 0 eV with DOS above threshold
            - estimate:
                  gap = CBM - VBM

    Suggested return:
        vbm, cbm, gap, threshold

    Note:
        This is an ESTIMATE from the DOS, not a uniquely exact method.
    """
    # "Empty" means DOS below 1% of the peak DOS (smearing never gives exactly 0).
    threshold = threshold_fraction * dos.max()
    filled = dos > threshold

    # Valence band maximum: highest energy at/below EF that still has states
    below = (energy_shifted <= 0) & filled
    vbm = energy_shifted[below].max()

    # Conduction band minimum: lowest energy above EF that has states again.
    # Start the search from the VBM so we jump over the gap.
    above = (energy_shifted > vbm) & filled
    # first energy above VBM with states, after at least one empty grid point
    empty_after_vbm = (energy_shifted > vbm) & ~filled
    if not empty_after_vbm.any():
        return vbm, vbm, 0.0, threshold        # no gap at all
    gap_start = energy_shifted[empty_after_vbm].min()
    cbm = energy_shifted[(energy_shifted > gap_start) & filled].min()

    gap = cbm - vbm
    return vbm, cbm, gap, threshold


def classify_material(dos_ef, dos_array):
    """
    Make a simple electronic classification from DOS(EF).

    TODO:
        Decide how to classify:
            - if DOS(EF) is clearly nonzero -> metal
            - if DOS(EF) is ~0 and a gap is visible -> semiconductor / insulator

    Return:
        classification string
    """
    # Compare DOS at EF to the typical size of the DOS.
    # Metal: states right at EF.  Semiconductor/insulator: essentially none.
    if dos_ef > 0.01 * dos_array.max():
        return "metal"
    return "semiconductor/insulator"


# ============================================================
# MAIN MATERIAL ANALYSIS
# ============================================================

def analyze_material(name, dos_directory):
    """
    Analyze one material (Al or Si).

    TODO:
        1. Read DOSCAR
        2. Read OUTCAR
        3. Get E_F
        4. Shift energy axis so EF = 0
        5. Compute DOS(EF)

    Return a dictionary with everything you need for plotting and summary.
    """
    dos_directory = Path(dos_directory)

    # 1. total DOS from DOSCAR
    energy, dos, efermi_header, n_atoms = read_doscar(dos_directory / "DOSCAR")
    # 2-3. Fermi energy from OUTCAR (what the assignment asks for)
    efermi = parse_fermi_from_outcar(dos_directory / "OUTCAR")
    # 4. put EF at 0 eV
    e_shift = shift_energy_axis(energy, efermi)
    # 5. DOS at EF, per cell and per atom (Al cell has 1 atom, Si cell has 8)
    dos_ef = dos_at_fermi(e_shift, dos)

    result = {
        "name": name,
        "efermi": efermi,
        "energy_shifted": e_shift,
        "dos": dos,
        "n_atoms": n_atoms,
        "dos_per_atom": dos / n_atoms,
        "dos_ef": dos_ef,
        "dos_ef_per_atom": dos_ef / n_atoms,
    }
    result["classification"] = classify_material(dos_ef, dos)
    if result["classification"] == "metal":
        # states at EF -> no gap to measure
        result.update(vbm=0.0, cbm=0.0, gap=0.0)
    else:
        vbm, cbm, gap, _ = estimate_gap_from_dos(e_shift, dos)
        result.update(vbm=vbm, cbm=cbm, gap=gap)
    return result


# ============================================================
# OUTPUT HELPERS
# ============================================================

def save_comparison_table(path, rows):
    """
    Save the final comparison table as CSV.
    """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "material",
            "fermi_energy_eV",
            "DOS_at_EF",
            "calculated_gap_eV",
            "classification",
        ])
        writer.writerows(rows)


# ============================================================
# MAIN WORKFLOW
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--al-dir", default="aluminum/dos",
                        help="Directory containing Al DOSCAR and OUTCAR")
    parser.add_argument("--si-dir", default="silicon/dos",
                        help="Directory containing Si DOSCAR and OUTCAR")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent

    # --------------------------------------------------------
    # 1. Analyze Al and Si
    # --------------------------------------------------------
    al = analyze_material("Al", root / args.al_dir)
    si = analyze_material("Si", root / args.si_dir)

    # --------------------------------------------------------
    # 2. Plot the DOS for both materials (per atom, so the 1-atom Al cell
    #    and the 8-atom Si cell are on the same footing)
    # --------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, m in zip(axes, [al, si]):
        ax.plot(m["energy_shifted"], m["dos_per_atom"], lw=1.2)
        ax.axvline(0, ls="--", lw=1, c="k", label=r"$E_F$")
        ax.set_xlim(-12, 8)
        ax.set_xlabel(r"$E - E_F$ (eV)")
        ax.set_title(f"{m['name']} ({m['classification']})")
    axes[0].set_ylabel("Total DOS (states/eV/atom)")

    # mark the Si band edges and shade the gap
    axes[1].axvline(si["vbm"], ls=":", c="tab:red", label=f"VBM {si['vbm']:.2f} eV")
    axes[1].axvline(si["cbm"], ls=":", c="tab:green", label=f"CBM {si['cbm']:.2f} eV")
    axes[1].axvspan(si["vbm"], si["cbm"], color="gray", alpha=0.2,
                    label=f"gap ≈ {si['gap']:.2f} eV")
    for ax in axes:
        ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(root / "problem5_Al_vs_Si_DOS.png", dpi=200)

    # --------------------------------------------------------
    # 3. Build the comparison table
    # --------------------------------------------------------
    rows = [
        ["Al", f"{al['efermi']:.4f}", f"{al['dos_ef_per_atom']:.4f}", 0.0, al["classification"]],
        ["Si", f"{si['efermi']:.4f}", f"{si['dos_ef_per_atom']:.4f}", f"{si['gap']:.3f}", si["classification"]],
    ]
    save_comparison_table(root / "problem5_comparison_table.csv", rows)

    # --------------------------------------------------------
    # 4. Print a concise summary to the screen
    # --------------------------------------------------------
    for m in (al, si):
        print(f"{m['name']}:")
        print(f"  Fermi energy (before shift): {m['efermi']:.4f} eV")
        print(f"  DOS(EF): {m['dos_ef_per_atom']:.4f} states/eV/atom "
              f"({m['dos_ef']:.4f} per {m['n_atoms']}-atom cell)")
        print(f"  VBM {m['vbm']:.3f} eV, CBM {m['cbm']:.3f} eV, gap {m['gap']:.3f} eV")
        print(f"  Classification: {m['classification']}")


if __name__ == "__main__":
    main()
