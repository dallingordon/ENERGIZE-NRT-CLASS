#!/usr/bin/env python3
"""
Pedagogical Si band-structure + DOS plot from the three-stage VASP workflow:

    01_scf/
    02_bands/
    03_dos/

Run from the TOP-LEVEL calculation directory:

    python plot_si_bands_dos_colored.py

Required:
    numpy
    matplotlib
    pymatgen

Color meaning:
    BLUE = valence bands / occupied-side DOS
    RED  = conduction bands / unoccupied-side DOS

The VBM and CBM are explicitly marked on the band structure.

If orbital-projected DOS is available in 03_dos/vasprun.xml (e.g. LORBIT=11),
the script also overlays Si s- and p-projected DOS.

Outputs:
    Si_bandstructure_DOS.png
    Si_bandstructure_DOS.pdf
"""

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

try:
    from pymatgen.io.vasp.outputs import Vasprun
    from pymatgen.electronic_structure.core import Spin, OrbitalType
except ImportError:
    raise SystemExit(
        "\nERROR: pymatgen is required.\n"
        "Install with:\n"
        "    pip install pymatgen matplotlib numpy\n"
    )

ENERGY_MIN = -8.0
ENERGY_MAX = 6.0
SHOW_PROJECTED_DOS = True

VALENCE_COLOR = "#1f77b4"
CONDUCTION_COLOR = "#d62728"
S_DOS_COLOR = "#2ca02c"
P_DOS_COLOR = "#9467bd"

ROOT = Path(__file__).resolve().parent

SCF_XML = ROOT / "01_scf" / "vasprun.xml"
BANDS_XML = ROOT / "02_bands" / "vasprun.xml"
BANDS_KPOINTS = ROOT / "02_bands" / "KPOINTS"
DOS_XML = ROOT / "03_dos" / "vasprun.xml"

required = [SCF_XML, BANDS_XML, BANDS_KPOINTS, DOS_XML]
missing = [p for p in required if not p.exists()]

if missing:
    print("\nERROR: missing required files:")
    for p in missing:
        print(f"  {p}")
    sys.exit(1)

print("Reading 01_scf/vasprun.xml ...")

scf_run = Vasprun(
    str(SCF_XML),
    parse_projected_eigen=False,
    parse_potcar_file=False,
)

gap_scf, cbm_scf, vbm_scf, is_direct = scf_run.eigenvalue_band_properties

print("\nSCF electronic structure")
print("------------------------")
print(f"VBM                  : {vbm_scf:10.4f} eV")
print(f"CBM                  : {cbm_scf:10.4f} eV")
print(f"PBE band gap         : {gap_scf:10.4f} eV")
print(f"Direct gap?          : {is_direct}")
print(f"SCF Fermi energy     : {scf_run.efermi:10.4f} eV")

print("\nReading 02_bands/vasprun.xml ...")

bands_run = Vasprun(
    str(BANDS_XML),
    parse_projected_eigen=False,
    parse_potcar_file=False,
)

band_structure = bands_run.get_band_structure(
    kpoints_filename=str(BANDS_KPOINTS),
    line_mode=True,
    efermi=scf_run.efermi,
)

k_distance = np.asarray(band_structure.distance, dtype=float)
bands = np.asarray(band_structure.bands[Spin.up], dtype=float)
bands_shifted = bands - vbm_scf
midgap = 0.5 * gap_scf

def clean_label(label):
    if label is None:
        return ""
    label = str(label)
    replacements = {
        "\\Gamma": "Γ",
        "$\\Gamma$": "Γ",
        "Gamma": "Γ",
        "GAMMA": "Γ",
    }
    return replacements.get(label, label)

raw_ticks = []

for distance, kpoint in zip(k_distance, band_structure.kpoints):
    label = getattr(kpoint, "label", None)
    if label:
        raw_ticks.append((float(distance), clean_label(label)))

tick_positions = []
tick_labels = []

for distance, label in raw_ticks:
    if tick_positions and np.isclose(distance, tick_positions[-1]):
        if label and label != tick_labels[-1]:
            if tick_labels[-1]:
                tick_labels[-1] = tick_labels[-1] + "|" + label
            else:
                tick_labels[-1] = label
    else:
        tick_positions.append(distance)
        tick_labels.append(label)

print("\nBand-path labels")
print("----------------")
for d, lab in zip(tick_positions, tick_labels):
    print(f"{d:10.5f}   {lab}")

valence_candidates = np.where(
    bands_shifted <= midgap,
    bands_shifted,
    -np.inf,
)

conduction_candidates = np.where(
    bands_shifted > midgap,
    bands_shifted,
    np.inf,
)

vbm_band_index, vbm_k_index = np.unravel_index(
    np.argmax(valence_candidates),
    valence_candidates.shape,
)

cbm_band_index, cbm_k_index = np.unravel_index(
    np.argmin(conduction_candidates),
    conduction_candidates.shape,
)

vbm_path_energy = bands_shifted[vbm_band_index, vbm_k_index]
cbm_path_energy = bands_shifted[cbm_band_index, cbm_k_index]

vbm_path_x = k_distance[vbm_k_index]
cbm_path_x = k_distance[cbm_k_index]

print("\nBand-path extrema")
print("-----------------")
print(f"VBM on plotted path : {vbm_path_energy:8.4f} eV")
print(f"CBM on plotted path : {cbm_path_energy:8.4f} eV")
print(f"Path gap            : {(cbm_path_energy-vbm_path_energy):8.4f} eV")

print("\nReading 03_dos/vasprun.xml ...")

dos_run = Vasprun(
    str(DOS_XML),
    parse_projected_eigen=False,
    parse_potcar_file=False,
)

tdos = dos_run.tdos

dos_energy = np.asarray(tdos.energies, dtype=float) - vbm_scf
dos_total = np.zeros_like(dos_energy)

for density in tdos.densities.values():
    dos_total += np.asarray(density, dtype=float)

s_dos = None
p_dos = None

if SHOW_PROJECTED_DOS:
    try:
        spd_dos = dos_run.complete_dos.get_spd_dos()

        if OrbitalType.s in spd_dos:
            s_dos = np.zeros_like(dos_energy)
            for density in spd_dos[OrbitalType.s].densities.values():
                s_dos += np.asarray(density, dtype=float)

        if OrbitalType.p in spd_dos:
            p_dos = np.zeros_like(dos_energy)
            for density in spd_dos[OrbitalType.p].densities.values():
                p_dos += np.asarray(density, dtype=float)

        if s_dos is not None or p_dos is not None:
            print("Projected DOS found: overlaying orbital character.")
        else:
            print(
                "Projected DOS not found. Showing total DOS only.\n"
                "For orbital-resolved DOS, use LORBIT = 11 in the DOS run."
            )

    except Exception as exc:
        print(
            "Projected DOS could not be read; showing total DOS only.\n"
            f"Reason: {exc}"
        )

fig, (ax_band, ax_dos) = plt.subplots(
    1,
    2,
    figsize=(11, 6.5),
    sharey=True,
    gridspec_kw={"width_ratios": [3.4, 1.35]},
)

valence_label_used = False
conduction_label_used = False

for band in bands_shifted:
    if np.nanmax(band) < midgap:
        label = "Valence bands" if not valence_label_used else None
        ax_band.plot(
            k_distance,
            band,
            color=VALENCE_COLOR,
            linewidth=1.15,
            label=label,
        )
        valence_label_used = True
    else:
        label = "Conduction bands" if not conduction_label_used else None
        ax_band.plot(
            k_distance,
            band,
            color=CONDUCTION_COLOR,
            linewidth=1.15,
            label=label,
        )
        conduction_label_used = True

ax_band.axhline(
    0.0,
    color="black",
    linestyle="--",
    linewidth=1.0,
)

ax_band.axhline(
    gap_scf,
    color="black",
    linestyle=":",
    linewidth=1.0,
)

ax_band.scatter(
    [vbm_path_x],
    [vbm_path_energy],
    s=65,
    color=VALENCE_COLOR,
    edgecolor="black",
    linewidth=0.8,
    zorder=10,
    label="VBM",
)

ax_band.scatter(
    [cbm_path_x],
    [cbm_path_energy],
    s=65,
    color=CONDUCTION_COLOR,
    edgecolor="black",
    linewidth=0.8,
    zorder=10,
    label="CBM",
)

for xpos in tick_positions:
    ax_band.axvline(
        xpos,
        color="0.80",
        linewidth=0.7,
        zorder=0,
    )

ax_band.set_xticks(tick_positions)
ax_band.set_xticklabels(tick_labels)
ax_band.set_xlim(k_distance.min(), k_distance.max())
ax_band.set_ylim(ENERGY_MIN, ENERGY_MAX)
ax_band.set_xlabel("Wave-vector path")
ax_band.set_ylabel("Energy relative to VBM (eV)")
ax_band.set_title("Si electronic band structure")

ax_band.text(
    0.03,
    0.96,
    f"PBE gap = {gap_scf:.3f} eV\n"
    f"{'Direct' if is_direct else 'Indirect'} gap",
    transform=ax_band.transAxes,
    ha="left",
    va="top",
)

ax_band.legend(
    loc="lower left",
    fontsize=8,
    frameon=True,
)

plot_mask = (
    (dos_energy >= ENERGY_MIN)
    & (dos_energy <= ENERGY_MAX)
)

valence_mask = plot_mask & (dos_energy <= 0.0)
conduction_mask = plot_mask & (dos_energy >= gap_scf)

ax_dos.plot(
    dos_total[valence_mask],
    dos_energy[valence_mask],
    color=VALENCE_COLOR,
    linewidth=1.5,
    label="Valence DOS",
)

ax_dos.plot(
    dos_total[conduction_mask],
    dos_energy[conduction_mask],
    color=CONDUCTION_COLOR,
    linewidth=1.5,
    label="Conduction DOS",
)

gap_mask = (
    plot_mask
    & (dos_energy > 0.0)
    & (dos_energy < gap_scf)
)

if np.any(gap_mask):
    ax_dos.plot(
        dos_total[gap_mask],
        dos_energy[gap_mask],
        color="0.5",
        linewidth=0.8,
    )

if s_dos is not None:
    ax_dos.plot(
        s_dos[plot_mask],
        dos_energy[plot_mask],
        color=S_DOS_COLOR,
        linestyle="--",
        linewidth=1.0,
        label="Si s",
    )

if p_dos is not None:
    ax_dos.plot(
        p_dos[plot_mask],
        dos_energy[plot_mask],
        color=P_DOS_COLOR,
        linestyle="--",
        linewidth=1.0,
        label="Si p",
    )

ax_dos.axhline(
    0.0,
    color="black",
    linestyle="--",
    linewidth=1.0,
)

ax_dos.axhline(
    gap_scf,
    color="black",
    linestyle=":",
    linewidth=1.0,
)

ax_dos.set_xlabel("DOS (states/eV)")
ax_dos.set_title("Si density of states")
ax_dos.set_xlim(left=0)
ax_dos.tick_params(axis="y", labelleft=False)

ax_dos.legend(
    loc="upper right",
    fontsize=8,
    frameon=True,
)

fig.suptitle(
    "Silicon: PBE Band Structure and Density of States",
    fontsize=14,
)

fig.tight_layout()

png_file = ROOT / "Si_bandstructure_DOS.png"
pdf_file = ROOT / "Si_bandstructure_DOS.pdf"

fig.savefig(
    png_file,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    pdf_file,
    bbox_inches="tight",
)

print("\nSaved:")
print(f"  {png_file}")
print(f"  {pdf_file}")

print("\nPlot color key")
print("--------------")
print("BLUE   = valence bands / occupied-side DOS")
print("RED    = conduction bands / unoccupied-side DOS")
print("GREEN  = Si s-projected DOS (if available)")
print("PURPLE = Si p-projected DOS (if available)")
print("Markers identify the VBM and CBM.")

print("\nInterpretation")
print("--------------")
print("Energy = 0 eV is the valence-band maximum.")
print(f"The calculated SCF PBE gap is {gap_scf:.3f} eV.")
print(
    f"The SCF calculation identifies the gap as "
    f"{'direct' if is_direct else 'indirect'}."
)
print(
    "The valence-band maximum and conduction-band minimum occur at "
    "different k-points for indirect-gap Si."
)

plt.show()
