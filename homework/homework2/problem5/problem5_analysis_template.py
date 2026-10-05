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
    raise NotImplementedError("TODO: implement parse_fermi_from_outcar()")


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
    raise NotImplementedError("TODO: implement read_doscar()")


# ============================================================
# DOS ANALYSIS HELPERS
# ============================================================

def shift_energy_axis(energy, efermi):
    """
    Shift energies so that EF = 0 eV.

    TODO:
        return energy - efermi
    """
    raise NotImplementedError("TODO: implement shift_energy_axis()")


def dos_at_fermi(energy_shifted, dos):
    """
    Estimate the DOS at EF = 0.

    TODO:
        Use interpolation to estimate DOS at 0 eV.

    Hint:
        np.interp(0.0, energy_shifted, dos)
    """
    raise NotImplementedError("TODO: implement dos_at_fermi()")


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
    raise NotImplementedError("TODO: implement estimate_gap_from_dos()")


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
    raise NotImplementedError("TODO: implement classify_material()")


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
    raise NotImplementedError("TODO: implement analyze_material()")


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
    # TODO:
    # al = analyze_material("Al", root / args.al_dir)
    # si = analyze_material("Si", root / args.si_dir)

    # --------------------------------------------------------
    # 2. Plot the DOS for both materials
    # --------------------------------------------------------
    # TODO:
    # Create either:
    #   - two separate panels, or
    #   - one combined plot with clearly separated curves
    #
    # Suggested structure:
    #
    # fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    #
    # For each material:
    #   axes[i].plot(energy_shifted, dos)
    #   axes[i].axvline(0, linestyle="--", linewidth=1, label=r"$E_F$")
    #   axes[i].set_xlabel(r"$E - E_F$ (eV)")
    #   axes[i].set_ylabel("Total DOS (states/eV)")
    #   axes[i].set_title("Material name")
    #
    # For Si, if you estimate VBM and CBM:
    #   mark them with vertical lines
    #
    # fig.tight_layout()
    # fig.savefig(root / "problem5_Al_vs_Si_DOS.png", dpi=200)

    # --------------------------------------------------------
    # 3. Build the comparison table
    # --------------------------------------------------------
    # TODO:
    # rows = [
    #     ["Al", al["efermi"], al["dos_ef"], 0.0, al["classification"]],
    #     ["Si", si["efermi"], si["dos_ef"], si["gap"], si["classification"]],
    # ]
    #
    # save_comparison_table(root / "problem5_comparison_table.csv", rows)

    # --------------------------------------------------------
    # 4. Print a concise summary to the screen
    # --------------------------------------------------------
    # TODO:
    # print("Al:")
    # print(f"  Fermi energy: {al['efermi']:.6f} eV")
    # print(f"  DOS(EF):      {al['dos_ef']:.6f}")
    # print(f"  Classification: {al['classification']}")
    #
    # print("Si:")
    # print(f"  Fermi energy: {si['efermi']:.6f} eV")
    # print(f"  DOS(EF):      {si['dos_ef']:.6f}")
    # print(f"  VBM:          {si['vbm']:.6f} eV")
    # print(f"  CBM:          {si['cbm']:.6f} eV")
    # print(f"  Gap:          {si['gap']:.6f} eV")
    # print(f"  Classification: {si['classification']}")

    print("Problem 5 template loaded.")
    print("Complete the TODO sections to perform the full DOS analysis.")


if __name__ == "__main__":
    main()
