"""Stage 3c: turn a score.py .npz into rankings. No model needed -- re-run freely with different options.

Options:
  --combine ensemble|repeats|both   mean over repeats (upstream add_ensemble), each repeat alone, or both
  --group   mineral|spectrum        best score per mineral (upstream reorganize_similarity_score = max over
                                    a mineral's reference spectra) or every reference spectrum ranked
  --topk N  --truth NAME            report top-N and the rank of the true mineral
  --csv PATH                        also write a table

Usage:
    python postprocess.py scores_eu.npz --combine both --group mineral --topk 5
"""
import argparse
import json
import os

import numpy as np


def rank_items(scores_1d, item_label, truth):
    order = np.argsort(scores_1d)[::-1]
    hit = np.where(item_label[order] == truth)[0] if truth is not None else []
    return order, (int(hit[0]) + 1 if len(hit) else -1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scores", help=".npz from score.py")
    ap.add_argument("--combine", choices=["ensemble", "repeats", "both"], default="ensemble")
    ap.add_argument("--group", choices=["mineral", "spectrum"], default="mineral")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--truth", default="silicon")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    S = np.load(a.scores, allow_pickle=False)
    scores, repeats, names = S["scores"], S["repeats"], S["input_name"]
    refs = np.load(os.path.join(str(S["cache_dir"]), "refs.npz"))
    tr_label, label_name, tr_file = refs["tr_label"], refs["label_name"], refs["tr_filename"]
    meta = json.loads(str(S["meta_json"]))

    t = [i for i, n in enumerate(label_name) if n.strip().lower() == a.truth.lower()] or \
        [i for i, n in enumerate(label_name) if a.truth.lower() in n.lower()]
    truth = t[0] if t else None
    print("model %s (%s) | %d refs, %d minerals | truth: %s"
          % (str(S["model"]), str(S["version"])[:40] + "...", meta["n_ref"], meta["n_minerals"],
             "%s (label %d)" % (label_name[truth], truth) if t else "NOT FOUND"))

    runs = []
    if a.combine in ("ensemble", "both"):
        runs.append(("ensemble", scores.mean(axis=0)))
    if a.combine in ("repeats", "both"):
        runs += [(str(r), scores[i]) for i, r in enumerate(repeats)]

    if a.group == "mineral":
        classes = np.unique(tr_label)
        idx = [np.where(tr_label == c)[0] for c in classes]
        item_label = classes
        item_name = [label_name[c] for c in classes]
    else:
        item_label = tr_label
        item_name = ["%s [%s]" % (label_name[l], f) for l, f in zip(tr_label, tr_file)]

    rows = []
    for run, sc in runs:
        if a.group == "mineral":
            sc = np.stack([sc[:, v].max(axis=1) for v in idx], axis=1)  # [input, mineral]
        print("\n=== %s | by %s | top-%d ===" % (run, a.group, a.topk))
        for n, s in zip(names, sc):
            order, rank = rank_items(s, item_label, truth)
            print("%-45s %s rank: %s" % (n, a.truth, rank))
            top = [(item_name[j], float(s[j])) for j in order[:a.topk]]
            for k, (m, v) in enumerate(top, 1):
                print("    %d. %-50s %.4f" % (k, m, v))
            rows.append([str(n), run, a.group, rank] + [x for p in top for x in (p[0], "%.4f" % p[1])])

    if a.csv:
        with open(a.csv, "w") as f:
            f.write(",".join(["input", "run", "group", "truth_rank"] +
                             ["top%d%s" % (k, s) for k in range(1, a.topk + 1) for s in ("", "_score")]) + "\n")
            for r in rows:
                f.write(",".join('"%s"' % x if "," in str(x) else str(x) for x in r) + "\n")
        print("\nwrote", a.csv)


if __name__ == "__main__":
    main()
