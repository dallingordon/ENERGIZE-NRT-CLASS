"""Plot each input spectrum against its nearest reference spectra (from a score.py .npz).

One panel per input: the input on top, then the top-k matches stacked below with an offset.
For --group mineral, each mineral is drawn with the specific reference spectrum that gave its best
score (the one the max in reorganize_similarity_score picked). Labels: rank, mineral, RRUFF file, score.

Usage:
    python plot_matches.py scores_eu.npz --topk 5 -o matches_eu.png
    python plot_matches.py scores_eu.npz --run repeat_2 --group spectrum --inputs si_sim
    python plot_matches.py scores_eu.npz --inputs si_sim --layout overlay   # sim drawn on top of each match
    python plot_matches.py scores_eu.npz --inputs si_sim --layout repeats   # #1 match of each model, overlaid
    python plot_matches.py scores_eu.npz --inputs si_sim --layout repeats --rank 2
    python plot_matches.py scores_eu.npz --layout models --group spectrum --topk 10   # every model, every ref spectrum
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def short_id(fname):
    """'Silicon__R040145__Raman__532__0__unoriented__...' -> 'R040145, 532 nm'"""
    parts = str(fname).split("__")
    if len(parts) >= 4 and parts[1].startswith("R"):
        return "%s, %s nm" % (parts[1], parts[3])
    return str(fname)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scores", help=".npz from score.py")
    ap.add_argument("--run", default="ensemble", help="'ensemble' (mean over repeats) or a repeat name, e.g. repeat_0")
    ap.add_argument("--group", choices=["mineral", "spectrum"], default="mineral")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--truth", default="silicon", help="highlight this mineral if it appears")
    ap.add_argument("--inputs", nargs="*", default=None, help="only plot inputs whose name contains one of these")
    ap.add_argument("--xlim", nargs=2, type=float, default=None)
    ap.add_argument("--layout", choices=["stack", "overlay", "repeats", "models"], default="stack",
                    help="stack: input on top, matches offset below; "
                         "overlay: one small panel per match with the input drawn on top of it; "
                         "repeats: one panel per input, the #1 match of the ensemble and of each of the "
                         "6 models drawn on top of each other (identical picks share a line); "
                         "models: grid, one row per input x one column per model, each a stacked top-k")
    ap.add_argument("--with-ensemble", action="store_true", help="repeats/models layouts: also include the ensemble mean")
    ap.add_argument("--rank", type=int, default=1, help="repeats layout: which rank to show (1 = top match)")
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

    def picks_for(s):
        if a.group == "mineral":
            best_ref = {c: members[c][np.argmax(s[members[c]])] for c in classes}
            m_score = np.array([s[best_ref[c]] for c in classes])
            return [best_ref[classes[j]] for j in np.argsort(m_score)[::-1][:a.topk]]
        return list(np.argsort(s)[::-1][:a.topk])

    def style(ax, xlabel=True):
        ax.set_yticks([])
        ax.set_xlim(a.xlim or (wn[0], wn[-1]))
        if xlabel:
            ax.set_xlabel("Raman shift (cm$^{-1}$)", fontsize=8)
        ax.tick_params(labelsize=7)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)

    if a.layout == "stack":
        fig, axes = plt.subplots(len(keep), 1, figsize=(9, len(keep) * (1.2 + 0.55 * (a.topk + 1))), squeeze=False)
        for ax, i in zip(axes[:, 0], keep):
            s = sc[i]
            step = 1.15
            ax.plot(wn, X[i] + step * a.topk, color="black", lw=1.4)
            ax.text(wn[-1], step * a.topk + 0.5, "input: %s" % names[i], ha="right", va="bottom",
                    fontsize=8, weight="bold")
            for k, r in enumerate(picks_for(s)):
                y0 = step * (a.topk - 1 - k)
                hit = str(label_name[tr_label[r]]).lower() == truth
                col = "#c0392b" if hit else "#2c6fbb"
                ax.plot(wn, tr_spec[r] + y0, color=col, lw=1.0)
                ax.text(wn[-1], y0 + 0.5, "%d. %s [%s]  %.2f" % (k + 1, label_name[tr_label[r]], short_id(tr_file[r]), s[r]),
                        ha="right", va="bottom", fontsize=7.5, color=col)
            style(ax)
    elif a.layout == "repeats":
        runs = ([("ensemble", scores.mean(axis=0))] if a.with_ensemble else []) + \
               [(r, scores[j]) for j, r in enumerate(repeats)]
        colors = plt.cm.tab10(np.arange(10))
        fig, axes = plt.subplots(len(keep), 1, figsize=(10, 3.4 * len(keep)), squeeze=False)
        for ax, i in zip(axes[:, 0], keep):
            ax.plot(wn, X[i], color="black", lw=2.2, label="input: %s" % names[i], zorder=10)
            picked = {}  # reference index -> list of (run, score)
            for run, sc_run in runs:
                r = picks_for(sc_run[i])[a.rank - 1]
                picked.setdefault(r, []).append((run, float(sc_run[i][r])))
            for c, (r, who) in enumerate(picked.items()):
                hit = str(label_name[tr_label[r]]).lower() == truth
                runs_txt = ", ".join("%s %.2f" % (w.replace("repeat_", "r"), v) for w, v in who)
                ax.plot(wn, tr_spec[r], color=colors[c % 10], lw=1.3, alpha=0.85,
                        ls="-" if hit else "--",
                        label="%s [%s]  <- %s" % (label_name[tr_label[r]], short_id(tr_file[r]), runs_txt))
            ax.set_title("%s | #%d match per model (solid = %s, dashed = other)" % (names[i], a.rank, a.truth),
                         fontsize=9)
            ax.legend(fontsize=7, loc="upper right", frameon=False)
            style(ax)
    elif a.layout == "models":
        runs = ([("ensemble", scores.mean(axis=0))] if a.with_ensemble else []) + \
               [(r, scores[j]) for j, r in enumerate(repeats)]
        step = 1.15
        fig, axes = plt.subplots(len(keep), len(runs), figsize=(3.6 * len(runs), (0.42 * (a.topk + 1) + 1.0) * len(keep)),
                                 squeeze=False, sharex=True)
        for row, i in enumerate(keep):
            for col, (run, sc_run) in enumerate(runs):
                ax = axes[row, col]
                s = sc_run[i]
                ax.plot(wn, X[i] + step * a.topk, color="black", lw=1.3)
                for k, r in enumerate(picks_for(s)):
                    y0 = step * (a.topk - 1 - k)
                    hit = str(label_name[tr_label[r]]).lower() == truth
                    c = "#c0392b" if hit else "#2c6fbb"
                    ax.plot(wn, tr_spec[r] + y0, color=c, lw=0.8)
                    ax.text(wn[-1], y0 + 0.45, "%d. %s %s %.2f" % (k + 1, label_name[tr_label[r]],
                                                                  short_id(tr_file[r]).split(",")[0], s[r]),
                            ha="right", va="bottom", fontsize=6, color=c)
                ax.set_title("%s\n%s" % (names[i], run) if col == 0 else run, fontsize=7.5, loc="left")
                style(ax, xlabel=(row == len(keep) - 1))
    else:
        ncol = min(a.topk, 3)
        nrow_per = int(np.ceil(a.topk / ncol))
        fig, axes = plt.subplots(len(keep) * nrow_per, ncol, figsize=(4.2 * ncol, 2.1 * nrow_per * len(keep)),
                                 squeeze=False)
        for b, i in enumerate(keep):
            s = sc[i]
            picks = picks_for(s)
            for k in range(nrow_per * ncol):
                ax = axes[b * nrow_per + k // ncol, k % ncol]
                if k >= len(picks):
                    ax.axis("off")
                    continue
                r = picks[k]
                hit = str(label_name[tr_label[r]]).lower() == truth
                col = "#c0392b" if hit else "#2c6fbb"
                ax.plot(wn, tr_spec[r], color=col, lw=1.0, label="%s [%s]" % (label_name[tr_label[r]], short_id(tr_file[r])))
                ax.plot(wn, X[i], color="black", lw=0.9, alpha=0.8, label=str(names[i]))
                ax.set_title("%s | #%d %s  score %.2f" % (names[i], k + 1, label_name[tr_label[r]], s[r]),
                             fontsize=7.5, color=col)
                ax.legend(fontsize=6, loc="upper right", frameon=False)
                style(ax, xlabel=(k // ncol == nrow_per - 1))
    if a.layout in ("repeats", "models"):
        fig.suptitle("%s model | each of the 6 models separately | top-%d by %s  (red = %s)"
                     % (S["model"], a.topk, a.group, a.truth), fontsize=9)
    else:
        fig.suptitle("%s model | %s | top-%d by %s  (red = %s)" % (S["model"], a.run, a.topk, a.group, a.truth),
                     fontsize=9)
    fig.tight_layout()
    out = a.out or "%s_%s_%s_%s.png" % (os.path.splitext(a.scores)[0], a.run, a.group, a.layout)
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
