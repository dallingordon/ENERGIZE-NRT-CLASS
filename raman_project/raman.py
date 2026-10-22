#!/usr/bin/env python3
"""
Raman spectrum from VASP by finite differences of the dielectric tensor along
Gamma-point phonon modes (same method as vasp_raman.py, Fonari & Stauffer).

Workflow (see run_pipeline.sh):
  1. relax                      -> 01_relax/CONTCAR
  2. DFPT phonons (IBRION=8, NWRITE=3) on the relaxed cell -> 02_phonon/OUTCAR
  3. python raman.py displace   -> 03_raman/mode_XX_{m,p}/POSCAR  (+/- step along each optical mode)
  4. LEPSILON run in every 03_raman/mode_* dir
  5. python raman.py collect    -> <out>_modes.csv, <out>_spectrum.csv, <out>.png

Only needs numpy (matplotlib optional for the plot).
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np


# ----------------------------------------------------------------------------- POSCAR I/O
def read_poscar(path):
    L = open(path).read().splitlines()
    scale = float(L[1].split()[0])
    raw = np.array([[float(x) for x in L[i].split()[:3]] for i in (2, 3, 4)])
    f = scale if scale > 0 else (-scale / abs(np.linalg.det(raw))) ** (1 / 3)
    lat = raw * f
    species = L[5].split()
    if not all(s[0].isalpha() for s in species):
        sys.exit(f"{path}: needs VASP5 format with a species line")
    counts = [int(x) for x in L[6].split()]
    i = 7
    if L[i].strip()[0] in "sS":  # Selective dynamics
        i += 1
    cartesian = L[i].strip()[0] in "cCkK"
    n = sum(counts)
    pos = np.array([[float(x) for x in L[i + 1 + j].split()[:3]] for j in range(n)])
    cart = pos * f if cartesian else pos @ lat
    return dict(comment=L[0].strip(), lat=lat, species=species, counts=counts, cart=cart)


def write_poscar(path, s, cart, comment=None):
    with open(path, "w") as fh:
        fh.write((comment or s["comment"]) + "\n1.0\n")
        for v in s["lat"]:
            fh.write("  %.10f %.10f %.10f\n" % tuple(v))
        fh.write("  " + " ".join(s["species"]) + "\n  " + " ".join(map(str, s["counts"])) + "\nCartesian\n")
        for p in cart:
            fh.write("  %.10f %.10f %.10f\n" % tuple(p))


# ----------------------------------------------------------------------------- OUTCAR parsing
MODE_RE = re.compile(r"^\s*(\d+)\s+f(/i)?\s*=.*?([-\d.]+)\s+cm-1")


def parse_modes(outcar, natoms):
    """Gamma modes from the 'Eigenvectors after division by SQRT(mass)' block (needs NWRITE=3)."""
    lines = open(outcar).read().splitlines()
    starts = [i for i, l in enumerate(lines) if "Eigenvectors after division by SQRT(mass)" in l]
    if not starts:
        sys.exit(f"{outcar}: no 'Eigenvectors after division by SQRT(mass)' block - set NWRITE=3 in the phonon run")
    modes, i = [], starts[-1]
    while i < len(lines) and len(modes) < 3 * natoms:
        m = MODE_RE.match(lines[i])
        if m:
            freq = float(m.group(3)) * (-1.0 if m.group(2) else 1.0)
            vec = np.array([[float(x) for x in lines[i + 2 + j].split()[3:6]] for j in range(natoms)])
            modes.append((freq, vec))
            i += 2 + natoms
        else:
            i += 1
    if len(modes) != 3 * natoms:
        sys.exit(f"{outcar}: parsed {len(modes)} modes, expected {3 * natoms}")
    return modes


def parse_eps(outcar):
    """Last 'MACROSCOPIC STATIC DIELECTRIC TENSOR (including local field effects in DFT)' (electronic eps_inf)."""
    lines = open(outcar).read().splitlines()
    key = "MACROSCOPIC STATIC DIELECTRIC TENSOR (including local field effects in DFT)"
    idx = [i for i, l in enumerate(lines) if l.strip() == key]
    if not idx:
        return None
    i = idx[-1]
    return np.array([[float(x) for x in lines[i + 2 + r].split()[:3]] for r in range(3)])


# ----------------------------------------------------------------------------- commands
def cmd_displace(a):
    s = read_poscar(a.poscar)
    modes = parse_modes(a.outcar, len(s["cart"]))
    os.makedirs(a.outdir, exist_ok=True)
    meta = []
    print(f"{'mode':>4} {'freq (cm-1)':>12}  action")
    for k, (freq, vec) in enumerate(modes, 1):
        if freq < a.min_freq:
            print(f"{k:4d} {freq:12.2f}  skipped (acoustic / below {a.min_freq} cm-1)")
            continue
        norm = float(np.linalg.norm(vec))
        for sign, tag in ((-1, "m"), (1, "p")):
            d = os.path.join(a.outdir, f"mode_{k:02d}_{tag}")
            os.makedirs(d, exist_ok=True)
            write_poscar(os.path.join(d, "POSCAR"), s, s["cart"] + sign * a.step * vec / norm,
                         comment=f"mode {k} {freq:.2f} cm-1 displaced {'+' if sign > 0 else '-'}{a.step} A")
        meta.append(dict(mode=k, freq=freq, norm=norm))
        print(f"{k:4d} {freq:12.2f}  -> mode_{k:02d}_m / _p")
    json.dump(dict(step=a.step, poscar=os.path.abspath(a.poscar), modes=meta),
              open(os.path.join(a.outdir, "modes.json"), "w"), indent=1)
    print(f"\n{2 * len(meta)} displaced structures written to {a.outdir}/")


def raman_activity(ra):
    alpha = (ra[0, 0] + ra[1, 1] + ra[2, 2]) / 3.0
    beta2 = ((ra[0, 0] - ra[1, 1]) ** 2 + (ra[0, 0] - ra[2, 2]) ** 2 + (ra[1, 1] - ra[2, 2]) ** 2
             + 6.0 * (ra[0, 1] ** 2 + ra[0, 2] ** 2 + ra[1, 2] ** 2)) / 2.0
    return 45.0 * alpha ** 2 + 7.0 * beta2


def cmd_collect(a):
    meta = json.load(open(os.path.join(a.outdir, "modes.json")))
    step = meta["step"]
    s = read_poscar(meta["poscar"])
    vol = abs(np.linalg.det(s["lat"]))
    rows, missing = [], []
    for m in meta["modes"]:
        k = m["mode"]
        eps = {}
        for tag in "mp":
            oc = os.path.join(a.outdir, f"mode_{k:02d}_{tag}", "OUTCAR")
            eps[tag] = parse_eps(oc) if os.path.isfile(oc) else None
            if eps[tag] is None:
                missing.append(oc)
        if eps["m"] is None or eps["p"] is None:
            continue
        ra = (eps["p"] - eps["m"]) * 0.5 / step * m["norm"] * vol / (4.0 * np.pi)
        rows.append(dict(mode=k, freq=m["freq"], activity=raman_activity(ra)))
    if missing:
        print("[warn] missing / unfinished dielectric runs:\n  " + "\n  ".join(missing))
    if not rows:
        sys.exit("no modes collected")

    # intensity: activity * (nu_L - nu)^4 / nu * (n_BE + 1)   (Stokes)
    nu_l = 1e7 / a.laser
    for r in rows:
        nu = r["freq"]
        n_be = 1.0 / np.expm1(1.4387769 * nu / a.temp) if a.temp > 0 else 0.0
        r["intensity"] = r["activity"] * (nu_l - nu) ** 4 / nu * (n_be + 1.0)
    imax = max(r["intensity"] for r in rows) or 1.0
    amax = max(r["activity"] for r in rows) or 1.0

    with open(a.out + "_modes.csv", "w") as fh:
        fh.write("mode,freq_cm-1,activity_A4_per_amu,activity_rel,intensity_rel\n")
        for r in rows:
            fh.write(f"{r['mode']},{r['freq']:.3f},{r['activity']:.6g},{r['activity'] / amax:.4f},"
                     f"{r['intensity'] / imax:.4f}\n")

    x = np.arange(a.xmin, a.xmax + 1e-9, a.dx)
    g = a.fwhm / 2.0
    y = np.zeros_like(x)
    for r in rows:
        y += (r["intensity"] / imax) * g ** 2 / ((x - r["freq"]) ** 2 + g ** 2)
    y /= y.max()
    np.savetxt(a.out + "_spectrum.csv", np.c_[x, y], delimiter=",", fmt="%.4f",
               header="wavenumber,intensity", comments="")

    print(f"{'mode':>4} {'freq':>10} {'activity':>12} {'I_rel':>7}")
    for r in rows:
        print(f"{r['mode']:4d} {r['freq']:10.2f} {r['activity']:12.4g} {r['intensity'] / imax:7.3f}")
    print(f"\nwrote {a.out}_modes.csv and {a.out}_spectrum.csv "
          f"({a.xmin}-{a.xmax} cm-1, Lorentzian FWHM {a.fwhm} cm-1, laser {a.laser} nm, T {a.temp} K)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.plot(x, y, lw=1.2)
        for r in rows:
            ax.axvline(r["freq"], color="grey", lw=0.5, ls=":")
        ax.set_xlabel("Raman shift (cm$^{-1}$)"); ax.set_ylabel("intensity (norm.)")
        ax.set_title(os.path.basename(a.out) + " - simulated Raman (VASP)")
        plt.tight_layout(); plt.savefig(a.out + ".png", dpi=130)
        print(f"wrote {a.out}.png")
    except ImportError:
        pass


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("displace", help="write +/- displaced POSCARs along each optical mode")
    d.add_argument("--outcar", default="02_phonon/OUTCAR")
    d.add_argument("--poscar", default="02_phonon/POSCAR")
    d.add_argument("--outdir", default="03_raman")
    d.add_argument("--step", type=float, default=0.01, help="displacement (A) along normalised mode")
    d.add_argument("--min_freq", type=float, default=20.0, help="skip modes below this (acoustic)")

    c = sub.add_parser("collect", help="dielectric derivatives -> Raman activities -> spectrum")
    c.add_argument("--outdir", default="03_raman")
    c.add_argument("--out", default="raman")
    c.add_argument("--laser", type=float, default=532.0, help="excitation wavelength (nm)")
    c.add_argument("--temp", type=float, default=300.0, help="K, for Bose factor (0 = off)")
    c.add_argument("--fwhm", type=float, default=6.0, help="Lorentzian FWHM (cm-1)")
    c.add_argument("--xmin", type=float, default=0.0)
    c.add_argument("--xmax", type=float, default=1600.0)
    c.add_argument("--dx", type=float, default=0.5)

    a = p.parse_args()
    {"displace": cmd_displace, "collect": cmd_collect}[a.cmd](a)


if __name__ == "__main__":
    main()
