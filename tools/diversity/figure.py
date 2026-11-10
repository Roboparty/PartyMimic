"""Data figure from analyze.py's outputs: (left) a t-SNE map of the 1-s training windows by source, (right) the share
of each held-out test set's windows that have a training window within tau, as the sources are added (data.json).

    uv run --with scikit-learn python tools/diversity/figure.py <work dir> <out dir of analyze.py> <out png>
"""
import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datacard.embed import fit_embedding  # noqa: E402
from datacard.features import FeatureSet  # noqa: E402

COLORS = {"lafan": "#1f77b4", "bones": "#9aa4ad", "kneel": "#2ca02c", "kungfu": "#d62728"}
PALETTE = ("#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22")
PER = 3000  # t-SNE windows per source (the largest source gets 3x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("data", help="analyze.py out dir (data.json)")
    ap.add_argument("out")
    a = ap.parse_args()
    d = json.load(open(Path(a.data) / "data.json"))
    fs = FeatureSet.load(Path(a.work) / "train_fs.npz")
    P = fit_embedding(fs, 32, seed=0, description="train").transform(fs.X)
    wg = fs.window_group.astype(str)
    src = list(d["sources"])
    col = {g: COLORS.get(g, PALETTE[i % len(PALETTE)]) for i, g in enumerate(src)}
    big = max(src, key=lambda g: d["sources"][g]["hours"])
    rng = np.random.default_rng(0)
    order = [big] + [g for g in src if g != big]  # the largest source is drawn first, underneath
    idx = np.concatenate([rng.choice(np.flatnonzero(wg == g), min(PER * (3 if g == big else 1), int((wg == g).sum())),
                                     replace=False) for g in order])
    Y = TSNE(2, perplexity=40, init="pca", random_state=0).fit_transform(P[idx])
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.25]})
    for g in order:
        m = wg[idx] == g
        s = d["sources"][g]
        ax.scatter(Y[m, 0], Y[m, 1], s=1.5 if g == big else 2.5, c=col[g], alpha=0.5 if g == big else 0.8,
                   linewidths=0, label=f"{s['title']}: {s['clips']:,} clips, {s['hours']:.1f} h", rasterized=True)
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(f"Training windows (1 s), t-SNE: {sum(s['hours'] for s in d['sources'].values()):.0f} h, "
                 f"{sum(s['clips'] for s in d['sources'].values()):,} clips", fontsize=10)
    ax.legend(loc="lower left", fontsize=7.5, markerscale=5, frameon=True)
    tests = list(d["test_clips"])
    mixes = list(d["mixes"].values())
    w = 0.8 / len(mixes)
    for i, (m, g) in enumerate(zip(mixes, src)):
        x = np.arange(len(tests)) + (i - (len(mixes) - 1) / 2) * w
        n = d["sources"][g]["title"] if i == 0 else "+ " + d["sources"][g]["title"]
        bx.bar(x, [100 * m["coverage"][t] for t in tests], w, color=col[g],
               label=f"{n} ({m['hours']:.1f} h, {m['effective_clusters']:.0f} motion types)")
    bx.set_xticks(range(len(tests)), [f"{t}\n({d['test_clips'][t]} clips)" for t in tests], fontsize=8.5)
    bx.set_ylabel("test windows with a close training window (%)", fontsize=8.5)
    bx.set_title("How new are the held-out test sets? (lower = more novel)", fontsize=10)
    bx.legend(fontsize=7.5, loc="upper right")
    bx.set_ylim(0, 75)
    bx.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
