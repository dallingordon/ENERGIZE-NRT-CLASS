"""Stage 3: match spectra against the RRUFF reference set with the trained
contrastive models (Li, Schmidt & Alstrom 2022), following test_experiment.ipynb.

For each model: load the reference (training) split the model was trained with,
score every input spectrum against every reference with each of the 6 repeats,
average the repeats (ensemble), collapse to one score per mineral, report top-k.

Usage (on the SCC, `conda activate raman`):
    python match_sim.py --model excellent_unoriented sim_inputs/*excellent_unoriented*.csv
    python match_sim.py --model raw sim_inputs/*_raw*.csv
    options: --topk N, --group mineral|spectrum, --per-repeat
Inputs are Stage-2 files (wavenumber,norm_intensity) on that model's grid.
"""
import argparse
import glob
import os
import sys

import numpy as np

DEFAULT_REPO = ("/projectnb/me500jc/students/dgordon/raman_matching/"
                "raman_spectra_matching_with_contrastive_learning")
# grid used to train each model; give_args_test() gets excellent_unoriented wrong (max_wave 1024)
GRIDS = {"raw": (40, 1496), "excellent_unoriented": (76, 1404)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="Stage-2 input CSVs for this model")
    ap.add_argument("--model", required=True, choices=sorted(GRIDS))
    ap.add_argument("--truth", default="silicon", help="mineral name to report the rank of (case-insensitive)")
    ap.add_argument("--topk", type=int, default=5, help="how many matches to report (default 5)")
    ap.add_argument("--group", choices=["mineral", "spectrum"], default="mineral",
                    help="mineral: best score per mineral (one row per mineral); "
                         "spectrum: rank every reference spectrum individually")
    ap.add_argument("--per-repeat", action="store_true",
                    help="also report each of the 6 trained repeats separately, not just their average")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--out", default=None, help="results CSV (default: match_<model>.csv in cwd)")
    args_cli = ap.parse_args()

    inputs = [os.path.abspath(p) for p in args_cli.inputs]
    missing = [p for p in inputs if not os.path.isfile(p)]
    if missing:  # fail before the slow reference loading
        sys.exit("input not found (run from the dir that has sim_inputs/?):\n  " + "\n  ".join(missing))
    out = os.path.abspath(args_cli.out or "match_%s_%s.csv" % (args_cli.model, args_cli.group))

    # the repo uses relative paths and package imports -> run from its root
    os.chdir(args_cli.repo)
    sys.path.insert(0, args_cli.repo)
    import const
    import test

    model_root = "../data/experiments/exp_time/%s/" % args_cli.model
    model_dir = model_root + [v for v in os.listdir(model_root) if "version_" in v][0]
    dir2load_data = model_dir + "/data_splitting/"
    ckpt_dir = [model_dir + "/" + v for v in sorted(os.listdir(model_dir)) if "repeat_" in v]
    dir2read_data = "../data/public_dataset/"

    args = const.give_args_test(raman_type=args_cli.model)
    args["min_wave"], args["max_wave"] = GRIDS[args_cli.model]
    args["pre_define_tt_filenames"] = True
    tr_data, _tt_data, _, label_name = test.get_data(args, dir2load_data, "cls", False, dir2read_data)
    tr_spectrum, tr_label = tr_data[0], np.asarray(tr_data[1])
    label_name = np.asarray(label_name)
    n_wave = np.shape(tr_spectrum)[1]
    ref_ids = ["ref%d" % i for i in range(len(tr_label))]
    fn_path = dir2load_data + "tr_filename.npy"
    if os.path.exists(fn_path):
        fn = np.load(fn_path, allow_pickle=True)
        if len(fn) == len(tr_label):
            ref_ids = [str(v) for v in fn]
        else:
            print("note: tr_filename.npy has %d entries vs %d references; using ref indices" % (len(fn), len(tr_label)))
    print("references: %d spectra, %d minerals, %d wavenumbers" % (len(tr_label), len(np.unique(tr_label)), n_wave))

    # our spectra
    names, spectra = [], []
    for p in inputs:
        d = np.loadtxt(p, delimiter=",", skiprows=1)
        if len(d) != n_wave:
            sys.exit("%s has %d points, model expects %d -- wrong grid?" % (p, len(d), n_wave))
        names.append(os.path.basename(p))
        spectra.append(d[:, 1])
    tt_spectrum = np.asarray(spectra, dtype=np.float32)

    # label for the true mineral (only used for the repo's accuracy printout + our rank column)
    truth_idx = [i for i, n in enumerate(label_name) if str(n).strip().lower() == args_cli.truth.lower()]
    if not truth_idx:
        truth_idx = [i for i, n in enumerate(label_name) if args_cli.truth.lower() in str(n).lower()]
    if not truth_idx:
        print("WARNING: '%s' not found in label_name; rank column will be empty" % args_cli.truth)
    truth = truth_idx[0] if truth_idx else int(tr_label[0])
    print("truth mineral: %s (label %d)" % (label_name[truth] if truth_idx else "?", truth))
    tt_label = np.full(len(tt_spectrum), truth)

    # ensemble over repeats (as in test_experiment.ipynb)
    string_use = ["ensemble_%d" % i for i in range(len(ckpt_dir))]
    prediction_g = {}
    for i, s_ckpt in enumerate(ckpt_dir):
        prediction, _, _ = test.get_model_baseon_modeldir(
            args, s_ckpt, [[tr_spectrum, tr_label], [tt_spectrum, tt_label], label_name], print_info=False)
        prediction_g[string_use[i]] = prediction
        print("  %s done" % os.path.basename(s_ckpt))
    group_index = [np.where(tr_label == i)[0] for i in np.unique(tt_label)]
    repeat_preds = {k: np.asarray(prediction_g[k]) for k in string_use}
    prediction_g, string_use = test.add_ensemble(prediction_g, string_use, False, group_index, tr_label, tt_label)

    runs = [("ensemble", np.asarray(prediction_g["ensemble_avg"]))]
    if args_cli.per_repeat:
        runs += [(k.replace("ensemble_", "repeat_"), v) for k, v in repeat_preds.items()]

    rows = []
    for run_name, pred in runs:
        print("\n=== %s model | %s | by %s | top-%d ===" % (args_cli.model, run_name, args_cli.group, args_cli.topk))
        if args_cli.group == "mineral":
            scores_all, _ = test.reorganize_similarity_score(pred, tr_label)
            item_label = np.unique(tr_label)                      # column -> mineral label
            item_name = [str(label_name[c]) for c in item_label]
        else:
            scores_all = pred[:, :len(tr_label)]                  # column -> reference spectrum
            item_label = tr_label
            item_name = ["%s [%s]" % (label_name[l], r) for l, r in zip(tr_label, ref_ids)]
        for n, scores in zip(names, scores_all):
            order = np.argsort(scores)[::-1]
            hit = np.where(item_label[order] == truth)[0] if truth_idx else []
            rank = int(hit[0]) + 1 if len(hit) else -1
            top = [(item_name[j], float(scores[j])) for j in order[:args_cli.topk]]
            print("%-45s %s rank: %s" % (n, args_cli.truth, rank))
            for k, (m, sc) in enumerate(top, 1):
                print("    %d. %-45s %.4f" % (k, m, sc))
            rows.append([n, run_name, rank] + [x for t in top for x in (t[0], "%.4f" % t[1])])

    with open(out, "w") as f:
        f.write(",".join(["input", "run", "truth_rank"] + ["top%d%s" % (k, s) for k in range(1, args_cli.topk + 1)
                                                           for s in ("", "_score")]) + "\n")
        for r in rows:
            f.write(",".join('"%s"' % x if "," in str(x) else str(x) for x in r) + "\n")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
