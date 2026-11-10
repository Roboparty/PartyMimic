"""Command line: ``python -m datacard <command>``.

Datasets are given as feature files (``.npz`` from ``features``), dataset specs (``.json``, see ``datasets.py``;
features are extracted and cached under ``--cache-dir``) or any4hdmi directories (``[GROUP=]PATH``). Several
values are concatenated into one dataset.

Examples::

    python -m datacard features --spec v7bones.json --out feats/v7bones.npz
    python -m datacard features --dir locomotion=/data/test/locomotion dance=/data/test/dance \\
        --name rp1_test --out feats/test.npz
    python -m datacard fit-reference --features feats/v7bones.npz feats/off_lafan.npz \\
        --description "v7bones + official LAFAN" --out-dir ref/
    python -m datacard card --dataset feats/v7bones.npz --target feats/test.npz --reference ref/ \\
        --out-dir cards/ --dedup auto [--dataset-mask feasible.npz]
    python -m datacard dedup --a feats/v7bones.npz --b feats/test.npz --out dups.json
    python -m datacard splits source --dataset feats/v7bones.npz --holdout-groups md ground \\
        --reference ref/ --target feats/test.npz --out split_source.json
    python -m datacard splits cluster --dataset feats/v7bones.npz --reference ref/ \\
        --target feats/test.npz --frac 0.1 --out split_cluster.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .card import Reference, build_card, fit_reference, write_card
from .datasets import DatasetSpec, spec_from_dirs
from .dedup import find_duplicates
from .features import FeatureSet, build_features
from .splits import add_novelty, clip_categories, cluster_heldout, recordings, source_heldout


def load_features(items: list[str], name: str | None = None, cache_dir: str | None = None,
                  workers: int = 16) -> FeatureSet:
    sets, dirs = [], []
    for it in items:
        path = it.rpartition("=")[2]
        if it.endswith(".npz"):
            sets.append(FeatureSet.load(it))
        elif it.endswith(".json"):
            spec = DatasetSpec.load(it)
            out = Path(cache_dir or "datacard_cache") / f"{spec.name}.npz"
            sets.append(build_features(spec, out, workers))
        elif os.path.isdir(path):
            dirs.append(it)
        else:
            raise FileNotFoundError(it)
    if dirs:
        spec = spec_from_dirs(name or Path(dirs[0].rpartition("=")[2]).name, dirs)
        out = Path(cache_dir or "datacard_cache") / f"{spec.name}.npz"
        sets.append(build_features(spec, out, workers))
    if len(sets) == 1:
        if name:
            sets[0].meta["name"] = name
        return sets[0]
    return FeatureSet.concat(sets, name or "+".join(s.name for s in sets))


def _write_json(obj, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
    print(f"wrote {path}")


def cmd_features(a):
    if a.spec:
        spec = DatasetSpec.load(a.spec)
    else:
        spec = spec_from_dirs(a.name or Path(a.dir[0].rpartition("=")[2]).name, a.dir, a.robot)
    if a.name:
        spec.name = a.name
    if a.robot:
        spec.robot = a.robot
    build_features(spec, a.out, a.workers)


def cmd_fit_reference(a):
    pool = load_features(a.features, a.pool_name, a.cache_dir, a.workers)
    ref = fit_reference(pool, a.description or pool.name, a.components, a.clusters, a.seed)
    ref.save(a.out_dir)
    print(f"wrote {a.out_dir}")


def _logo_extra(items):
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        out[k] = v.split(",")
    return out


def cmd_card(a):
    ds = load_features(a.dataset, a.dataset_name, a.cache_dir, a.workers)
    tg = load_features(a.target, a.target_name, a.cache_dir, a.workers)
    ref = Reference.load(a.reference)
    dedup = None
    if a.dedup == "auto":
        dedup = find_duplicates(ds, tg, a.tol, a.min_len_s, a.metric, workers=a.workers)
    elif a.dedup not in (None, "none"):
        with open(a.dedup) as f:
            dedup = json.load(f)
    card, arrays = build_card(ds, tg, ref, a.k_radius, a.k_novelty, a.tau, a.dataset_mask, a.target_mask,
                              a.mask_missing, dedup, _logo_extra(a.logo_extra), a.least)
    paths = write_card(card, arrays, a.out_dir, a.stem or f"card_{ds.name}_vs_{tg.name}")
    print("\n".join(f"wrote {p}" for p in paths.values()))


def cmd_dedup(a):
    A = load_features(a.a, a.a_name, a.cache_dir, a.workers)
    B = load_features(a.b, a.b_name, a.cache_dir, a.workers)
    r = find_duplicates(A, B, a.tol, a.min_len_s, a.metric, a.partial_min_len_s, a.partial_min_motion,
                        cand_factor=a.cand_factor, min_window_motion=a.min_window_motion, workers=a.workers)
    _write_json(r, a.out)
    whole = sorted(n for n, v in r["b_clips"].items() if v["whole_a_matches"])
    print(f"{len(whole)} {B.name} clips contain a whole {A.name} clip; {len(r['b_clips']) - len(whole)} more overlap "
          f"partially; {len(r['matches'])} matches")
    for n in whole:
        print(f"  {n}  <-  {r['b_clips'][n]['whole_a_clips'][0]}")


def _categories(a, ds):
    if a.reference and a.target and a.categories == "target":
        ref = Reference.load(a.reference)
        tg = load_features(a.target, None, a.cache_dir, a.workers)
        P, Pt = ref.embedding.embed(ds), ref.embedding.embed(tg)
        units = np.unique(recordings(ds), return_inverse=True)[1]
        return clip_categories(ds, P, Pt, tg.window_group, "target", units), ref, P
    return clip_categories(ds, by="group"), (Reference.load(a.reference) if a.reference else None), None


def cmd_splits(a):
    ds = load_features(a.dataset, a.dataset_name, a.cache_dir, a.workers)
    cats, ref, P = _categories(a, ds)
    if a.kind == "source":
        out, held, cat_clip = source_heldout(ds, a.holdout_groups, a.name_regex, cats)
    else:
        if ref is None:
            raise SystemExit("splits cluster needs --reference")
        P = ref.embedding.embed(ds) if P is None else P
        coarse = ref.codebook.coarse_map(a.bins, a.seed)
        labels = coarse[ref.codebook.assign(P)]
        out, held, cat_clip = cluster_heldout(ds, labels, int(coarse.max()) + 1, cats, a.frac, a.threshold, a.seed)
    if ref is not None:
        P = ref.embedding.embed(ds) if P is None else P
        add_novelty(out, ds, P, held, cat_clip, a.seed)
    _write_json(out, a.out)
    s = out["summary"]
    print(f"train {s['train']['hours']:.1f} h / {s['train']['clips']} clips, held-out {s['heldout']['hours']:.1f} h / "
          f"{s['heldout']['clips']} clips")
    for c, v in s["per_category"].items():
        print(f"  {c:14s} held-out {v['heldout_hours']:6.2f} h ({v['heldout_frac']:.3f}) {v['heldout_clips']} clips")


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m datacard", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--cache-dir", default=None, help="feature cache for spec / directory inputs (default ./datacard_cache)")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("features", help="extract window features of a dataset")
    g = f.add_mutually_exclusive_group(required=True)
    g.add_argument("--spec")
    g.add_argument("--dir", nargs="+", help="any4hdmi directories, [GROUP=]PATH")
    f.add_argument("--name")
    f.add_argument("--robot")
    f.add_argument("--out", required=True)
    f.set_defaults(fn=cmd_features)

    r = sub.add_parser("fit-reference", help="fit the frozen embedding + codebook on a reference pool")
    r.add_argument("--features", nargs="+", required=True)
    r.add_argument("--pool-name")
    r.add_argument("--description", default="")
    r.add_argument("--components", type=int, default=32)
    r.add_argument("--clusters", type=int, default=2000)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out-dir", required=True)
    r.set_defaults(fn=cmd_fit_reference)

    c = sub.add_parser("card", help="data card of a dataset vs a target set")
    c.add_argument("--dataset", nargs="+", required=True)
    c.add_argument("--dataset-name")
    c.add_argument("--target", nargs="+", required=True)
    c.add_argument("--target-name")
    c.add_argument("--reference", required=True)
    c.add_argument("--out-dir", required=True)
    c.add_argument("--stem")
    c.add_argument("--dataset-mask")
    c.add_argument("--target-mask")
    c.add_argument("--mask-missing", choices=("keep", "drop"), default="keep")
    c.add_argument("--k-radius", type=int, default=5)
    c.add_argument("--k-novelty", type=int, default=10)
    c.add_argument("--tau", type=float, default=None, help="fixed radius (default: the reference tau)")
    c.add_argument("--dedup", default="auto", help="auto | none | duplicates.json from the dedup command")
    c.add_argument("--tol", type=float, default=0.05)
    c.add_argument("--min-len-s", type=float, default=1.0)
    c.add_argument("--metric", choices=("mean", "max"), default="mean")
    c.add_argument("--logo-extra", nargs="*", help="extra leave-out sets NAME=g1,g2")
    c.add_argument("--least", type=int, default=10)
    c.set_defaults(fn=cmd_card)

    d = sub.add_parser("dedup", help="duplicated segments between two datasets (raw joint space)")
    d.add_argument("--a", nargs="+", required=True)
    d.add_argument("--a-name")
    d.add_argument("--b", nargs="+", required=True)
    d.add_argument("--b-name")
    d.add_argument("--tol", type=float, default=0.05, help="joint tolerance in rad")
    d.add_argument("--min-len-s", type=float, default=1.0, help="minimum whole-segment length")
    d.add_argument("--metric", choices=("mean", "max"), default="mean",
                   help="mean: mean |dq| over the segment (prototype rule); max: every |dq| <= tol")
    d.add_argument("--partial-min-len-s", type=float, default=3.0)
    d.add_argument("--partial-min-motion", type=float, default=1.0, help="rad")
    d.add_argument("--cand-factor", type=float, default=1.6, help="candidate radius in units of tol per block mean")
    d.add_argument("--min-window-motion", type=float, default=0.1)
    d.add_argument("--out", required=True)
    d.set_defaults(fn=cmd_dedup)

    s = sub.add_parser("splits", help="source- or cluster-held-out splits without source leakage")
    s.add_argument("kind", choices=("source", "cluster"))
    s.add_argument("--dataset", nargs="+", required=True)
    s.add_argument("--dataset-name")
    s.add_argument("--reference")
    s.add_argument("--target", nargs="+")
    s.add_argument("--categories", choices=("target", "group"), default="target")
    s.add_argument("--holdout-groups", nargs="*", default=["md", "ground", "getup"])
    s.add_argument("--name-regex", default=None, help="also hold out clips whose name matches (source split)")
    s.add_argument("--frac", type=float, default=0.1)
    s.add_argument("--threshold", type=float, default=0.7, help="family cut, Hellinger distance (cluster split)")
    s.add_argument("--bins", type=int, default=128, help="coarse codebook size for family histograms")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_splits)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
