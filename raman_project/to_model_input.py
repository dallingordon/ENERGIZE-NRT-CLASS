"""Stage 2: convert a Raman spectrum CSV into the input format of a trained
contrastive-matching model (Li, Schmidt & Alstrom 2022).

Uses the network repo's own interpolation + normalization (data/rruff.py:
find_interp, norm) so our spectra are treated exactly like the RRUFF references.

Usage (on the SCC):
    python to_model_input.py SPECTRUM.csv --model raw
    python to_model_input.py SPECTRUM.csv --model excellent_unoriented --scale 1.036
Writes <stem>_<model>[_x<scale>]_input.csv next to the input (or to --out).
"""
import argparse
import os
import sys

import numpy as np

DEFAULT_REPO = ("/projectnb/me500jc/students/dgordon/raman_matching/"
                "raman_spectra_matching_with_contrastive_learning")

# min_wave, max_wave as used to train each model (grid = np.arange(max_wave)[min_wave:]).
# Set here explicitly: const.give_args_test() overwrites excellent_unoriented's max_wave with 1024.
GRIDS = {
    "raw": (40, 1496),                   # 1456 points, 40-1495 cm-1
    "excellent_unoriented": (76, 1404),  # 1328 points, 76-1403 cm-1
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spectrum", help="CSV with a header and two columns: wavenumber, intensity")
    ap.add_argument("--model", required=True, choices=sorted(GRIDS))
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply wavenumbers by this before interpolating (e.g. 1.036 for PBE softening)")
    ap.add_argument("--repo", default=DEFAULT_REPO, help="path to the network repo (for data.rruff)")
    ap.add_argument("--out", default=None, help="output CSV path")
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    import data.rruff as rruff  # noqa: E402  (from the network repo)

    spec = np.loadtxt(args.spectrum, delimiter=",", skiprows=1)  # [N, 2]
    spec = spec[np.argsort(spec[:, 0])]
    spec[:, 0] *= args.scale

    min_wave, max_wave = GRIDS[args.model]
    grid = np.arange(max_wave)[min_wave:]
    y = rruff.find_interp(spec, grid)          # linear, zero outside the data range
    y = rruff.norm(y, "max_1_min_0")

    out = args.out
    if out is None:
        stem = os.path.splitext(args.spectrum)[0]
        for suffix in ("_raw", "_spectrum"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
        tag = "" if args.scale == 1.0 else "_x%g" % args.scale
        out = "%s_%s%s_input.csv" % (stem, args.model, tag)
    np.savetxt(out, np.column_stack([grid, y]), delimiter=",", fmt="%.4f",
               header="wavenumber,norm_intensity", comments="")
    print("%s -> %s  (%d points, %d-%d cm-1, peak at %d cm-1)"
          % (args.spectrum, out, len(grid), grid[0], grid[-1], grid[np.argmax(y)]))


if __name__ == "__main__":
    main()
