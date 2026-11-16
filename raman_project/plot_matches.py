"""Plot each input spectrum against its nearest reference spectra (from a score.py .npz).

One panel per input: the input on top, then the top-k matches stacked below with an offset.
For --group mineral, each mineral is drawn with the specific reference spectrum that gave its best
score (the one the max in reorganize_similarity_score picked). Labels: rank, mineral, RRUFF file, score.

Usage:
    python plot_matches.py scores_eu.npz --topk 5 -o matches_eu.png
    python plot_matches.py scores_eu.npz --run repeat_2 --group spectrum --inputs si_sim
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scores", help=".npz from score.py")
    ap.add_argument("--run", default="ensemble", help="'ensemble' (mean over repeats) or a repeat name, e.g. repeat_0")
    ap.add_argument("--group", choices=["mineral", "spectrum"], default="mineral")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--truth", default="silicon", help="highlight this mineral if it appears")
    ap.add_argument("--inputs", nargs="*", default=None, help="only plot inputs whose name contains one of these")
    ap.add_argument("--xlim", nargs=2, type=float, default=None)
    ap.add_argument("-o", "--out", default=None, help="output PNG (default: <scores>_<run>_<group>.png)")
    a = ap.parse_args()

    S = np.load(a.scores, allow_pickle=False)
    scores, repeats, names, X = S["scores"], [str(r) for r in S["repeats"]], S["input_name"], S["input_spectra"]
    refs = np.load(os.path.join(str(S["cache_dir"]), "refs.npz"))
    tr_spec, tr_label, label_name, tr_file, wn = (refs["tr_spectrum"], refs["tr_label"], refs["label_name"],
                                                  refs["tr_filename"], refs["wavenumber"])

    if a.run == "ensemble":
        sc = scores.mean(axis=0)
    elif a.run in repeats:
        sc = scores[repeats.index(a.run)]
    else:
        raise SystemExit("--run must be 'ensemble' or one of %s" % repeats)

    keep = [i for i, n in enumerate(names) if not a.inputs or any(s in str(n) for s in a.inputs)]
    if not keep:
        raise SystemExit("no inputs match %s" % a.inputs)

    classes = np.unique(tr_label)
    members = {c: np.where(tr_label == c)[0] for c in classes}
    truth = a.truth.lower()

    fig, axes = plt.subplots(len(keep), 1, figsize=(9, len(keep) * (1.2 + 0.55 * (a.topk + 1))),
                             squeeze=False)
    for ax, i in zip(axes[:, 0], keep):
        s = sc[i]
        if a.group == "mineral":
            best_ref = {c: members[c][np.argmax(s[members[c]])] for c in classes}
            m_score = np.array([s[best_ref[c]] for c in classes])
            picks = [best_ref[classes[j]] for j in np.argsort(m_score)[::-1][:a.topk]]
        else:
            picks = list(np.argsort(s)[::-1][:a.topk])

        step = 1.15
        ax.plot(wn, X[i] + step * a.topk, color="black", lw=1.4)
        ax.text(wn[-1], step * a.topk + 0.5, "input: %s" % names[i], ha="right", va="bottom", fontsize=8, weight="bold")
        for k, r in enumerate(picks):
            y0 = step * (a.topk - 1 - k)
            hit = str(label_name[tr_label[r]]).lower() == truth
            col = "#c0392b" if hit else "#2c6fbb"
            ax.plot(wn, tr_spec[r] + y0, color=col, lw=1.0)
            ax.text(wn[-1], y0 + 0.5, "%d. %s [%s]  %.2f" % (k + 1, label_name[tr_label[r]], tr_file[r], s[r]),
                    ha="right", va="bottom", fontsize=7.5, color=col)
        ax.set_yticks([])
        ax.set_xlim(a.xlim or (wn[0], wn[-1]))
        ax.set_xlabel("Raman shift (cm$^{-1}$)", fontsize=8)
        ax.tick_params(labelsize=7)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("%s model | %s | top-%d by %s  (red = %s)" % (S["model"], a.run, a.topk, a.group, a.truth),
                 fontsize=9)
    fig.tight_layout()
    out = a.out or "%s_%s_%s.png" % (os.path.splitext(a.scores)[0], a.run, a.group)
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
