"""Size and diversity of a training library, with the data-card features (datacard/: heading-invariant 1-s kinematic
windows, PCA embedding, k-means codebook):

* per source: clips, hours, occupied motion clusters and their effective number (exp entropy), reference speed;
* for the cumulative mixes of the sources in the given order: effective clusters and the coverage of every held-out
  test set (share of test windows within tau of a training window, tau = the reference's median cross-source 5-NN
  distance);
* <out>/data.json, and <work>/train_fs.npz, embed2d.npz for figure.py.

    uv run --with scikit-learn python tools/diversity/analyze.py <work dir> <out dir> \\
        --source lafan:LAFAN1=data/motions/g1/lafan --source bones:BONES-SEED=data/motions/g1/bones \\
        --source kungfu:KungfuAthlete=data/motions/g1/kungfu \\
        --md <sim2real>/datasets/motiondecode --test Extreme=data/testsets/extreme_g1

Sources are directories of training npz clips (tools/data/prep_motions.py output), tests any4hdmi datasets
(manifest.json + motions/); --md adds the four MotionDecode splits (Loco, Mani, Dance, Ground).
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datacard.card import fit_reference  # noqa: E402
from datacard.codebook import occupancy  # noqa: E402
from datacard.coverage import nn_distances  # noqa: E402
from datacard.datasets import DatasetSpec, GroupSpec  # noqa: E402
from datacard.features import DIFFICULTY_KEYS, build_features  # noqa: E402

MD_SPLITS = (("Loco", "locomotion"), ("Mani", "manipulation"), ("Dance", "dance"), ("Ground", "ground"))


def to_qpos(args):
    src, dst = args
    if not os.path.exists(dst):
        z = np.load(src)
        q = np.concatenate([z["body_pos_w"][:, 0], z["body_quat_w"][:, 0], z["joint_pos"]], 1)
        np.savez(dst, qpos=q.astype(np.float32))


def parse_source(s: str) -> tuple[str, str, Path]:
    key, path = s.split("=", 1)
    name, _, title = key.partition(":")
    return name, title or name, Path(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("work")
    ap.add_argument("out")
    ap.add_argument("--source", action="append", required=True, help="name[:title]=<dir of training npz>, in mix order")
    ap.add_argument("--md", default=None, help="sim2real datasets/motiondecode")
    ap.add_argument("--test", action="append", default=[], help="name=<any4hdmi dataset dir>")
    ap.add_argument("--robot", default="g1_29dof")
    ap.add_argument("--k", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    work, out = Path(a.work), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sources = [parse_source(s) for s in a.source]
    jobs, groups = [], []
    for g, _, d in sources:
        (work / g).mkdir(parents=True, exist_ok=True)
        jobs += [(str(f), str(work / g / f.name)) for f in sorted(d.glob("*.npz"))]
        groups.append(GroupSpec(g, motions=str(work / g), timestep=0.02))
    with Pool(a.workers) as p:
        p.map(to_qpos, jobs, chunksize=64)
    train = build_features(DatasetSpec("train", groups, robot=a.robot), work / "train_fs.npz", workers=a.workers)
    tg = []
    if a.md:
        tg += [GroupSpec(t, files=sorted(str(f) for f in (Path(a.md) / sub).rglob("*.npz")), timestep=1 / 120)
               for t, sub in MD_SPLITS]
    for t in a.test:
        name, path = t.split("=", 1)
        tg.append(GroupSpec(name, dir=path))
    if not tg:
        ap.error("no test sets: pass --md and/or --test")
    test = build_features(DatasetSpec("heldout", tg, robot=a.robot), work / "test_fs.npz", workers=a.workers)

    ref = fit_reference(train, "train", k=a.k)
    P, Pt = ref.embedding.embed(train), ref.embedding.embed(test)
    lab = ref.codebook.assign(P)
    wg, tcat = train.window_group.astype(str), test.window_group.astype(str)
    stride = train.meta["stride_s"]
    res = {"tau": ref.tau, "k": a.k, "explained_variance": ref.meta["explained_variance"], "sources": {}, "mixes": {}}
    cg = train.clip_group.astype(str)
    for g, title, _ in sources:
        c = cg == g
        occ, _ = occupancy(lab[wg == g], a.k, stride_s=stride)
        res["sources"][g] = {"title": title, "clips": int(c.sum()), "hours": float(train.clip_hours()[c].sum()),
                             "effective_clusters": occ["effective_clusters"],
                             **{f"{k}_median": float(np.median(train.diff[c, j])) for j, k in enumerate(DIFFICULTY_KEYS)},
                             **{f"{k}_p90": float(np.percentile(train.diff[c, j], 90)) for j, k in enumerate(DIFFICULTY_KEYS)}}
    tests = [g.group for g in tg]
    tclip = test.clip
    for i in range(len(sources)):
        keep = [g for g, _, _ in sources[:i + 1]]
        m = np.isin(wg, keep)
        occ, _ = occupancy(lab[m], a.k, stride_s=stride)
        d1 = nn_distances(Pt, P[m], 1)[:, 0]
        cov = {t: float(np.mean(d1[tcat == t] <= ref.tau)) for t in tests}
        res["mixes"]["+".join(keep)] = {"hours": float(train.clip_hours()[np.isin(cg, keep)].sum()),
                                        "clips": int(np.isin(cg, keep).sum()),
                                        "effective_clusters": occ["effective_clusters"],
                                        "occupied_clusters": occ["occupied_clusters"], "coverage": cov,
                                        "d1_median": {t: float(np.median(d1[tcat == t])) for t in tests}}
    res["test_hours"] = {t: float(test.clip_hours()[test.clip_group.astype(str) == t].sum()) for t in tests}
    res["test_clips"] = {t: int(np.sum(test.clip_group.astype(str) == t)) for t in tests}
    json.dump(res, open(out / "data.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    np.savez(work / "embed2d.npz", P=P[:, :2], g=wg, Pt=Pt[:, :2], t=tcat, tclip=tclip)


if __name__ == "__main__":
    main()
