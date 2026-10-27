#!/usr/bin/env python3
"""
Analyze the FCC Al NPT -> NVT -> NVE LAMMPS exercise.

Expected files in the same directory:
    out
    al_trajectory.dump

The script creates one five-panel figure:
  (a) mean atomic speed +/- 1 standard deviation vs timestep
  (b) mean force magnitude +/- 1 standard deviation vs timestep
  (c) total, kinetic, and potential energy per atom vs timestep
  (d) mean-squared displacement vs timestep, with an NVE diffusion fit
  (e) time-averaged RDF for NPT, NVT, and NVE

It also prints the self-diffusion coefficient obtained from the NVE segment:
    MSD(t) = 6 D t + b

Notes
-----
* LAMMPS "metal" units are used:
    velocity -> Angstrom/ps
    force    -> eV/Angstrom
    time     -> ps
    energy   -> eV
* The diffusion coefficient is fitted only in NVE because the NPT stage changes
  the box size. The final result is reported in both A^2/ps and cm^2/s.
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT_FILE = ROOT / "out"
DUMP_FILE = ROOT / "al_trajectory.dump"
FIG_FILE = ROOT / "al_md_analysis.png"
SUMMARY_FILE = ROOT / "al_md_summary.txt"

# Use the second half of the NVE segment for the linear diffusion fit.
NVE_FIT_FRACTION = 0.50
RDF_BIN_WIDTH_A = 0.05


# =====================================================================
# Read phase markers and thermo data from "out"
# =====================================================================
def parse_out(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    phase_starts = {}
    phase_ends = {}
    thermo_rows = []
    thermo_columns = None
    active_columns = None

    phase_start_re = re.compile(r"PHASE_START\s+(\w+)\s+(\d+)")
    phase_end_re = re.compile(r"PHASE_END\s+(\w+)\s+(\d+)")

    with path.open(errors="ignore") as f:
        for raw in f:
            line = raw.strip()

            m = phase_start_re.search(line)
            if m:
                phase_starts[m.group(1)] = int(m.group(2))

            m = phase_end_re.search(line)
            if m:
                phase_ends[m.group(1)] = int(m.group(2))

            parts = line.split()
            if not parts:
                continue

            # LAMMPS prints a new thermo header at the start of each run.
            if parts[0] == "Step" and "TotEng" in parts and "PotEng" in parts:
                active_columns = parts
                if thermo_columns is None:
                    thermo_columns = parts
                continue

            if active_columns is not None:
                if line.startswith("Loop time") or line.startswith("Performance"):
                    active_columns = None
                    continue

                if len(parts) == len(active_columns):
                    try:
                        vals = [float(x) for x in parts]
                    except ValueError:
                        continue
                    thermo_rows.append(dict(zip(active_columns, vals)))

    if not thermo_rows:
        raise RuntimeError(
            "No LAMMPS thermo data were found in 'out'. Make sure stdout was "
            "redirected to a file named 'out'."
        )

    required_phases = ["NPT", "NVT", "NVE"]
    if not all(p in phase_starts and p in phase_ends for p in required_phases):
        raise RuntimeError(
            "Could not find all PHASE_START/PHASE_END markers in 'out'. "
            "Use the supplied LAMMPS input without removing those print lines."
        )

    thermo = pd.DataFrame(thermo_rows)
    # Boundary timesteps can appear at the end of one run and the beginning of
    # the next. Keep one row per timestep.
    thermo = thermo.sort_values("Step").drop_duplicates("Step", keep="last")

    phases = {
        p: (phase_starts[p], phase_ends[p])
        for p in required_phases
    }
    return thermo, phases


# =====================================================================
# Read custom LAMMPS dump
# =====================================================================
def read_dump(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    frames = []
    with path.open() as f:
        while True:
            line = f.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue

            step = int(f.readline().strip())

            if not f.readline().startswith("ITEM: NUMBER OF ATOMS"):
                raise RuntimeError("Unexpected dump format near NUMBER OF ATOMS")
            natoms = int(f.readline().strip())

            box_header = f.readline().strip()
            if not box_header.startswith("ITEM: BOX BOUNDS"):
                raise RuntimeError("Unexpected dump format near BOX BOUNDS")

            bounds = []
            for _ in range(3):
                lo, hi, *rest = map(float, f.readline().split())
                bounds.append((lo, hi))
            box = np.array([hi - lo for lo, hi in bounds], dtype=float)

            atom_header = f.readline().strip().split()[2:]
            col = {name: i for i, name in enumerate(atom_header)}

            required = [
                "id", "x", "y", "z", "xu", "yu", "zu",
                "vx", "vy", "vz", "fx", "fy", "fz"
            ]
            missing = [name for name in required if name not in col]
            if missing:
                raise RuntimeError(f"Dump is missing columns: {missing}")

            data = np.empty((natoms, len(atom_header)), dtype=float)
            for i in range(natoms):
                data[i] = np.fromstring(f.readline(), sep=" ")

            # Ensure a consistent atom ordering across frames.
            order = np.argsort(data[:, col["id"]].astype(int))
            data = data[order]

            frames.append({
                "step": step,
                "natoms": natoms,
                "box": box,
                "wrapped": data[:, [col["x"], col["y"], col["z"]]],
                "unwrapped": data[:, [col["xu"], col["yu"], col["zu"]]],
                "velocity": data[:, [col["vx"], col["vy"], col["vz"]]],
                "force": data[:, [col["fx"], col["fy"], col["fz"]]],
            })

    if not frames:
        raise RuntimeError("No trajectory frames were read from the dump file.")

    natoms0 = frames[0]["natoms"]
    if any(fr["natoms"] != natoms0 for fr in frames):
        raise RuntimeError("Number of atoms changed during the trajectory.")

    return frames


def phase_for_step(step, phases):
    # Treat the boundary frame as belonging to the stage that starts there,
    # except the final endpoint.
    if phases["NPT"][0] <= step < phases["NPT"][1]:
        return "NPT"
    if phases["NVT"][0] <= step < phases["NVT"][1]:
        return "NVT"
    if phases["NVE"][0] <= step <= phases["NVE"][1]:
        return "NVE"
    return None


# =====================================================================
# Per-frame velocity and force statistics
# =====================================================================
def trajectory_statistics(frames):
    rows = []
    for fr in frames:
        speed = np.linalg.norm(fr["velocity"], axis=1)
        force_mag = np.linalg.norm(fr["force"], axis=1)
        rows.append({
            "step": fr["step"],
            "speed_mean": speed.mean(),
            "speed_std": speed.std(ddof=1),
            "force_mean": force_mag.mean(),
            "force_std": force_mag.std(ddof=1),
        })
    return pd.DataFrame(rows)


# =====================================================================
# MSD and self-diffusion coefficient
# =====================================================================
def compute_phase_local_msd(frames, phases, timestep_ps):
    """
    Compute an MSD referenced to the first saved frame of EACH ensemble.

    This keeps NPT box rescaling from being carried into the later fixed-box
    segments. For each frame, center-of-mass translation is removed before
    calculating MSD.
    """
    result = []
    per_phase = {}

    for phase in ["NPT", "NVT", "NVE"]:
        phase_frames = [fr for fr in frames if phase_for_step(fr["step"], phases) == phase]
        if len(phase_frames) < 2:
            continue

        r0 = phase_frames[0]["unwrapped"].copy()
        step0 = phase_frames[0]["step"]
        phase_rows = []

        for fr in phase_frames:
            dr = fr["unwrapped"] - r0
            # Remove any residual center-of-mass translation.
            dr -= dr.mean(axis=0, keepdims=True)
            msd = np.mean(np.sum(dr * dr, axis=1))
            time_from_phase_start_ps = (fr["step"] - step0) * timestep_ps
            row = {
                "step": fr["step"],
                "phase": phase,
                "time_from_phase_start_ps": time_from_phase_start_ps,
                "msd_A2": msd,
            }
            result.append(row)
            phase_rows.append(row)

        per_phase[phase] = pd.DataFrame(phase_rows)

    return pd.DataFrame(result), per_phase


def fit_nve_diffusion(nve_msd):
    if nve_msd is None or len(nve_msd) < 5:
        raise RuntimeError("Not enough NVE MSD points for a diffusion fit.")

    # Ignore the first portion of NVE so the fit emphasizes the later-time
    # diffusive behavior rather than the short-time ballistic regime.
    start_index = int(np.floor(len(nve_msd) * NVE_FIT_FRACTION))
    fit_df = nve_msd.iloc[start_index:].copy()

    t = fit_df["time_from_phase_start_ps"].to_numpy()
    y = fit_df["msd_A2"].to_numpy()

    slope, intercept = np.polyfit(t, y, 1)
    D_A2_per_ps = slope / 6.0
    # 1 A^2/ps = 1e-4 cm^2/s
    D_cm2_per_s = D_A2_per_ps * 1.0e-4

    fit_df["msd_fit_A2"] = slope * t + intercept
    return D_A2_per_ps, D_cm2_per_s, slope, intercept, fit_df


# =====================================================================
# RDF
# =====================================================================
def rdf_for_phase(frames, phase, phases, r_edges):
    """Time-average g(r) using wrapped positions and minimum-image distances."""
    hist_total = np.zeros(len(r_edges) - 1, dtype=float)
    norm_total = np.zeros(len(r_edges) - 1, dtype=float)
    shell_volume = (4.0 / 3.0) * np.pi * (r_edges[1:]**3 - r_edges[:-1]**3)

    phase_frames = [fr for fr in frames if phase_for_step(fr["step"], phases) == phase]
    if not phase_frames:
        raise RuntimeError(f"No dump frames found for {phase}.")

    for fr in phase_frames:
        pos = fr["wrapped"]
        box = fr["box"]
        n = len(pos)
        volume = np.prod(box)

        dists = []
        for i in range(n - 1):
            dr = pos[i + 1:] - pos[i]
            dr -= box * np.rint(dr / box)
            dists.append(np.sqrt(np.sum(dr * dr, axis=1)))
        dists = np.concatenate(dists)

        hist, _ = np.histogram(dists, bins=r_edges)
        hist_total += hist

        # Expected number of unordered pairs in each shell for an ideal gas.
        npairs = n * (n - 1) / 2.0
        norm_total += npairs * shell_volume / volume

    g_r = hist_total / norm_total
    r_centers = 0.5 * (r_edges[1:] + r_edges[:-1])
    return r_centers, g_r


# =====================================================================
# Plot helpers
# =====================================================================
def shade_phases(ax, phases):
    # Colors chosen only for visual separation of ensembles.
    colors = {"NPT": "#f5b7b1", "NVT": "#aed6f1", "NVE": "#abebc6"}
    for phase in ["NPT", "NVT", "NVE"]:
        start, end = phases[phase]
        ax.axvspan(start, end, alpha=0.24, color=colors[phase], zorder=0)
        mid = 0.5 * (start + end)
        ax.text(
            mid, 0.93, phase,
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontweight="bold"
        )
    for boundary in [phases["NPT"][1], phases["NVT"][1]]:
        ax.axvline(boundary, linestyle="--", linewidth=1.0, color="0.35")


# =====================================================================
# Main analysis
# =====================================================================
thermo, phases = parse_out(OUT_FILE)
frames = read_dump(DUMP_FILE)
stats = trajectory_statistics(frames)

natoms = frames[0]["natoms"]

# Determine timestep size from the thermo Time column when available.
if "Time" in thermo.columns and len(thermo) >= 2:
    dt_candidates = np.diff(thermo["Time"].to_numpy()) / np.diff(thermo["Step"].to_numpy())
    dt_candidates = dt_candidates[np.isfinite(dt_candidates) & (dt_candidates > 0)]
    timestep_ps = float(np.median(dt_candidates))
else:
    timestep_ps = 0.001

msd_all, msd_by_phase = compute_phase_local_msd(frames, phases, timestep_ps)
D_A2_ps, D_cm2_s, msd_slope, msd_intercept, nve_fit_df = fit_nve_diffusion(
    msd_by_phase.get("NVE")
)

if D_A2_ps <= 0:
    print(
        "WARNING: the fitted NVE MSD slope is non-positive. The trajectory does "
        "not show a clear diffusive regime, so the fitted self-diffusion "
        "coefficient should be interpreted as approximately zero / not resolved."
    )

# RDF cutoff must be below half the shortest instantaneous box dimension.
min_half_box = min(np.min(fr["box"]) for fr in frames) * 0.5
r_max = 0.98 * min_half_box
r_edges = np.arange(0.0, r_max + RDF_BIN_WIDTH_A, RDF_BIN_WIDTH_A)

rdf = {}
for phase in ["NPT", "NVT", "NVE"]:
    rdf[phase] = rdf_for_phase(frames, phase, phases, r_edges)

# Energy per atom.
energy = thermo.copy()
for col in ["PotEng", "KinEng", "TotEng"]:
    if col not in energy.columns:
        raise RuntimeError(f"Expected thermo column '{col}' was not found in out.")
energy["PE_per_atom"] = energy["PotEng"] / natoms
energy["KE_per_atom"] = energy["KinEng"] / natoms
energy["Etot_per_atom"] = energy["TotEng"] / natoms

# ---------------------------------------------------------------------
# Five-panel figure
# ---------------------------------------------------------------------
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.08], hspace=0.34, wspace=0.34)

ax_v = fig.add_subplot(gs[0, 0:2])
ax_f = fig.add_subplot(gs[0, 2:4])
ax_e = fig.add_subplot(gs[0, 4:6])
ax_m = fig.add_subplot(gs[1, 0:3])
ax_r = fig.add_subplot(gs[1, 3:6])

# Velocity magnitude distribution.
shade_phases(ax_v, phases)
ax_v.plot(stats["step"], stats["speed_mean"], label="Mean |v|")
ax_v.fill_between(
    stats["step"],
    np.maximum(0.0, stats["speed_mean"] - stats["speed_std"]),
    stats["speed_mean"] + stats["speed_std"],
    alpha=0.25,
    label="± 1 std. dev.",
)
ax_v.set_title("(a) Atomic speed distribution")
ax_v.set_xlabel("Timestep")
ax_v.set_ylabel("Speed (Å/ps)")
ax_v.legend(fontsize=9)

# Force magnitude distribution.
shade_phases(ax_f, phases)
ax_f.plot(stats["step"], stats["force_mean"], label="Mean |F|")
ax_f.fill_between(
    stats["step"],
    np.maximum(0.0, stats["force_mean"] - stats["force_std"]),
    stats["force_mean"] + stats["force_std"],
    alpha=0.25,
    label="± 1 std. dev.",
)
ax_f.set_title("(b) Atomic force distribution")
ax_f.set_xlabel("Timestep")
ax_f.set_ylabel("Force magnitude (eV/Å)")
ax_f.legend(fontsize=9)

# Energies.
shade_phases(ax_e, phases)
ax_e.plot(energy["Step"], energy["Etot_per_atom"], label="Total")
ax_e.plot(energy["Step"], energy["KE_per_atom"], label="Kinetic")
ax_e.plot(energy["Step"], energy["PE_per_atom"], label="Potential")
ax_e.set_title("(c) Energies per atom")
ax_e.set_xlabel("Timestep")
ax_e.set_ylabel("Energy (eV/atom)")
ax_e.legend(fontsize=9)

# MSD. Each phase is referenced to its own first saved frame.
shade_phases(ax_m, phases)
for phase in ["NPT", "NVT", "NVE"]:
    dfp = msd_by_phase.get(phase)
    if dfp is not None:
        ax_m.plot(dfp["step"], dfp["msd_A2"], label=f"{phase} MSD")

# Convert NVE fit times back to global timestep for overlay.
nve_start_step = msd_by_phase["NVE"].iloc[0]["step"]
fit_global_step = nve_start_step + nve_fit_df["time_from_phase_start_ps"] / timestep_ps
ax_m.plot(
    fit_global_step,
    nve_fit_df["msd_fit_A2"],
    linestyle="--",
    linewidth=2,
    label="NVE linear fit",
)
ax_m.set_title("(d) Mean-squared displacement")
ax_m.set_xlabel("Timestep")
ax_m.set_ylabel("MSD (Å$^2$)")
ax_m.legend(fontsize=9)
ax_m.text(
    0.98, 0.06,
    f"D(NVE) = {D_A2_ps:.3e} Å²/ps\n= {D_cm2_s:.3e} cm²/s",
    transform=ax_m.transAxes,
    ha="right", va="bottom",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
)

# RDF.
for phase in ["NPT", "NVT", "NVE"]:
    r, g = rdf[phase]
    ax_r.plot(r, g, label=phase)
ax_r.set_title("(e) Time-averaged radial distribution function")
ax_r.set_xlabel("r (Å)")
ax_r.set_ylabel("g(r)")
ax_r.set_xlim(0, r_max)
ax_r.legend()

fig.suptitle("FCC Al molecular dynamics: NPT → NVT → NVE", fontsize=18, fontweight="bold")
fig.savefig(FIG_FILE, dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------
# Save useful processed tables and summary.
# ---------------------------------------------------------------------
stats.to_csv(ROOT / "velocity_force_statistics.csv", index=False)
energy.to_csv(ROOT / "energy_vs_timestep.csv", index=False)
msd_all.to_csv(ROOT / "msd_vs_timestep.csv", index=False)

rdf_table = pd.DataFrame({"r_A": rdf["NPT"][0]})
for phase in ["NPT", "NVT", "NVE"]:
    rdf_table[f"g_r_{phase}"] = rdf[phase][1]
rdf_table.to_csv(ROOT / "rdf_by_ensemble.csv", index=False)

summary = f"""FCC Al NPT -> NVT -> NVE analysis
=================================
Number of atoms: {natoms}
LAMMPS timestep: {timestep_ps:.6f} ps

Phase boundaries (timesteps)
NPT: {phases['NPT'][0]} -> {phases['NPT'][1]}
NVT: {phases['NVT'][0]} -> {phases['NVT'][1]}
NVE: {phases['NVE'][0]} -> {phases['NVE'][1]}

NVE self-diffusion fit
MSD slope: {msd_slope:.8e} A^2/ps
D = slope / 6
D = {D_A2_ps:.8e} A^2/ps
D = {D_cm2_s:.8e} cm^2/s

The NVE fit uses the final {100*(1-NVE_FIT_FRACTION):.0f}% of the NVE MSD data.
If the 1000 K configuration remains crystalline, D may be very small; that is a physically meaningful result.
"""
SUMMARY_FILE.write_text(summary)

print(summary)
print("Created:")
print(f"  {FIG_FILE.name}")
print("  velocity_force_statistics.csv")
print("  energy_vs_timestep.csv")
print("  msd_vs_timestep.csv")
print("  rdf_by_ensemble.csv")
print(f"  {SUMMARY_FILE.name}")
