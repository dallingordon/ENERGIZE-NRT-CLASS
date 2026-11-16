"""Stage 3b: score input spectra against every cached reference with every repeat. No aggregation.

Writes one .npz with the full-granularity result:
  scores[repeat, input, reference]  similarity from upstream model.forward_test_batch
  + input names/paths/sha256/spectra, repeat names, cache dir, a copy of the cache meta
Post-process with postprocess.py.

Usage (repo-independent paths; run anywhere):
    python score.py --model excellent_unoriented sim_inputs/*excellent_unoriented*.csv -o scores_eu.npz
    python score.py ... --verify   # also run upstream's own path for repeat_0 and compare (slow)
"""
import argparse
import datetime
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rmatch_common as rc  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="Stage-2 input CSVs (wavenumber,norm_intensity) on the model grid")
    ap.add_argument("--model", required=True, choices=sorted(rc.GRIDS))
    ap.add_argument("-o", "--out", required=True, help="output .npz")
    ap.add_argument("--repo", default=rc.DEFAULT_REPO)
    ap.add_argument("--cache", default=rc.DEFAULT_CACHE)
    ap.add_argument("--verify", action="store_true",
                    help="cross-check repeat_0 against upstream get_model_baseon_modeldir (recomputes refs; slow)")
    a = ap.parse_args()

    inputs = [os.path.abspath(p) for p in a.inputs]
    missing = [p for p in inputs if not os.path.isfile(p)]
    if missing:
        sys.exit("input not found:\n  " + "\n  ".join(missing))
    out = os.path.abspath(a.out)

    rc.enter_repo(a.repo)
    import torch
    import test

    model_dir, version, repeats = rc.model_paths(a.model)
    cdir = rc.cache_dir(a.cache, a.model, version)
    meta = rc.load_meta(cdir)
    refs = np.load(os.path.join(cdir, "refs.npz"))
    wavenumber = refs["wavenumber"]

    # inputs: must sit exactly on the cached grid
    spectra = []
    for p in inputs:
        d = np.loadtxt(p, delimiter=",", skiprows=1)
        if len(d) != len(wavenumber) or not np.allclose(d[:, 0], wavenumber):
            sys.exit("%s is not on the %s grid (%d-%d cm-1, %d pts)"
                     % (p, a.model, wavenumber[0], wavenumber[-1], len(wavenumber)))
        spectra.append(d[:, 1])
    spectra = np.asarray(spectra, dtype=np.float32)

    scores, stale = [], []
    x = rc.to_tensor(torch, spectra, test.device)
    for r in repeats:
        info = meta["repeats"][r]
        st = os.stat(info["ckpt"])
        if st.st_size != info["ckpt_bytes"] or st.st_mtime != info["ckpt_mtime"]:
            stale.append(r)  # cheap check; sha256 recorded in meta for a full check
        model = rc.build_model(test, len(wavenumber), model_dir + "/" + r)
        feats = torch.load(os.path.join(cdir, "%s_features.pt" % r)).to(test.device)
        t0 = time.time()
        with torch.no_grad():
            s = model.forward_test_batch(x, feats)  # upstream: encodes the inputs, compares to all refs
        s = s.detach().cpu().numpy()[:, :len(refs["tr_label"])]
        scores.append(s)
        print("  %s scored %s [%.1fs]" % (r, s.shape, time.time() - t0))
    if stale:
        print("WARNING: checkpoint changed since cache was built for %s -> rebuild the cache" % stale)
    scores = np.asarray(scores, dtype=np.float32)  # [repeat, input, reference]

    if a.verify:
        tr = [refs["tr_spectrum"], refs["tr_label"]]
        tt = [spectra, np.zeros(len(spectra), dtype=np.int32)]
        ref_pred, _, _ = test.get_model_baseon_modeldir(
            {"dataset": "RRUFF"}, model_dir + "/" + repeats[0], [tr, tt, refs["label_name"]], print_info=False)
        diff = np.abs(np.asarray(ref_pred)[:, :scores.shape[2]] - scores[0]).max()
        print("verify repeat_0 vs upstream path: max |diff| = %.3g" % diff)

    np.savez_compressed(out, scores=scores, repeats=np.array(repeats), input_name=np.array([os.path.basename(p) for p in inputs]),
                        input_path=np.array(inputs), input_sha256=np.array([rc.sha256(p) for p in inputs]),
                        input_spectra=spectra, cache_dir=cdir, model=a.model, version=version,
                        meta_json=json.dumps(meta), stale=np.array(stale),
                        created=datetime.datetime.now().isoformat(timespec="seconds"))
    print("wrote", out, "scores", scores.shape)


if __name__ == "__main__":
    main()
