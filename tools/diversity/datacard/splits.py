"""Held-out splits of a training dataset without source leakage.

* Source-held-out: hold out whole groups (e.g. the MotionDecode-derived ``md`` / ``ground``; ``getup`` is a hard-linked
  subset of ``ground``), optionally plus clips of other groups whose name matches a provenance regex.
* Cluster-held-out: clips are merged into recording units (``datasets.recording_id``: ``name#k`` segments, frame-range
  chunks ``__000000_000256`` and ``_M`` mirrors of one recording form one unit), units are grouped into motion
  families by average-linkage agglomerative clustering of their coarse codebook histograms (Hellinger distance,
  within each category), and whole families are held out in random (seeded) order until ``frac`` of the category's
  hours is held out (families that would overshoot ``frac * (1 + slack)`` are skipped).

In both, every clip sharing a recording unit with a held-out clip is held out too, so segments of one source never
straddle the split. Categories are the dataset groups or, with a target set, the target category nearest to each unit
(majority over its windows of the category of the nearest target window). ``split_novelty`` reports how far the
held-out windows are from the training side, compared with a random unit-level split of the same size.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np

from .datasets import recording_id
from .features import FeatureSet
from .knn import knn


def recordings(fs: FeatureSet, patterns: tuple[str, ...] | None = None) -> np.ndarray:
    return np.array([recording_id(str(s), patterns) for s in fs.clip_source])


def clip_categories(fs: FeatureSet, P: np.ndarray | None = None, P_target: np.ndarray | None = None,
                    target_cat: np.ndarray | None = None, by: str = "target", units: np.ndarray | None = None,
                    device: str | None = None) -> np.ndarray:
    """Category per clip: its group, or the majority nearest-target-window category (pooled over its recording unit)."""
    if by == "group":
        return fs.clip_group.astype(str).copy()
    if P is None or P_target is None or target_cat is None:
        raise ValueError("target categories need the dataset / target embeddings and target categories")
    _, idx = knn(P, P_target, 1, device=device)
    cats, tc = np.unique(np.asarray(target_cat).astype(str), return_inverse=True)
    units = np.arange(fs.n_clips) if units is None else units
    votes = np.zeros((int(units.max()) + 1, len(cats)))
    np.add.at(votes, (units[fs.clip], tc[idx[:, 0]]), 1.0)
    groups, gi = np.unique(fs.clip_group.astype(str), return_inverse=True)
    gvotes = np.zeros((len(groups), len(cats)))
    np.add.at(gvotes, gi, votes[units])
    v = votes[units]
    empty = v.sum(1) == 0
    v[empty] = gvotes[gi[empty]]
    out = cats[v.argmax(1)].astype(object)
    out[v.sum(1) == 0] = fs.clip_group[v.sum(1) == 0]
    return out.astype(str)


def _clip_rows(fs: FeatureSet, idx: np.ndarray, rec: np.ndarray, cat: np.ndarray) -> list[dict]:
    return [{"name": str(fs.clip_name[i]), "path": str(fs.clip_path[i]), "group": str(fs.clip_group[i]),
             "source": str(fs.clip_source[i]), "recording": str(rec[i]), "category": str(cat[i])} for i in idx]


def _sizes(fs: FeatureSet, mask: np.ndarray, key: np.ndarray, rec: np.ndarray) -> dict:
    h = fs.clip_hours()
    out = {}
    for k in np.unique(key):
        m = key == k
        out[str(k)] = {"train_hours": float(h[m & ~mask].sum()), "heldout_hours": float(h[m & mask].sum()),
                       "train_clips": int((m & ~mask).sum()), "heldout_clips": int((m & mask).sum()),
                       "train_recordings": int(len(set(rec[m & ~mask]))),
                       "heldout_recordings": int(len(set(rec[m & mask])))}
        tot = out[str(k)]["train_hours"] + out[str(k)]["heldout_hours"]
        out[str(k)]["heldout_frac"] = out[str(k)]["heldout_hours"] / tot if tot > 0 else 0.0
    return out


def _check_no_leak(held: np.ndarray, rec: np.ndarray, src: np.ndarray) -> dict:
    leak_rec = set(rec[held]) & set(rec[~held])
    leak_src = set(src[held]) & set(src[~held])
    if leak_rec or leak_src:
        raise AssertionError(f"split leaks {len(leak_rec)} recordings / {len(leak_src)} sources across sides")
    return {"shared_recordings": 0, "shared_sources": 0}


def _propagate(held: np.ndarray, rec: np.ndarray) -> tuple[np.ndarray, int]:
    out = held | np.isin(rec, np.unique(rec[held]))
    return out, int((out & ~held).sum())


def _finish(kind: str, fs: FeatureSet, held: np.ndarray, rec: np.ndarray, cat: np.ndarray, params: dict,
            extra: dict | None = None) -> dict:
    leak = _check_no_leak(held, rec, fs.clip_source.astype(str))
    h = fs.clip_hours()
    summary = {"train": {"clips": int((~held).sum()), "hours": float(h[~held].sum()), "recordings": len(set(rec[~held]))},
               "heldout": {"clips": int(held.sum()), "hours": float(h[held].sum()), "recordings": len(set(rec[held]))},
               "per_category": _sizes(fs, held, cat, rec), "per_group": _sizes(fs, held, fs.clip_group.astype(str), rec),
               "leakage": leak, **(extra or {})}
    return {"kind": kind, "dataset": fs.name, "params": params, "summary": summary,
            "heldout_recordings": sorted(set(rec[held].tolist())),
            "train": _clip_rows(fs, np.flatnonzero(~held), rec, cat),
            "heldout": _clip_rows(fs, np.flatnonzero(held), rec, cat)}


def source_heldout(fs: FeatureSet, holdout_groups=(), name_regex: str | None = None, categories: np.ndarray | None = None,
                   patterns: tuple[str, ...] | None = None) -> tuple[dict, np.ndarray, np.ndarray]:
    """Hold out whole groups (and clips whose name matches ``name_regex``), plus every clip of their recordings.

    Returns (split dict, held-out clip mask, category per clip).
    """
    rec = recordings(fs, patterns)
    cat = fs.clip_group.astype(str) if categories is None else categories
    held = np.isin(fs.clip_group, list(holdout_groups))
    by_name = 0
    if name_regex:
        rx = re.compile(name_regex)
        m = np.array([bool(rx.search(str(n))) for n in fs.clip_name])
        by_name = int((m & ~held).sum())
        held |= m
    held, moved = _propagate(held, rec)
    params = {"holdout_groups": list(holdout_groups), "name_regex": name_regex,
              "recording_patterns": list(patterns) if patterns is not None else "default"}
    out = _finish("source_heldout", fs, held, rec, cat, params,
                  {"added_by_name_regex": by_name, "added_by_shared_recording": moved})
    return out, held, np.asarray(cat).astype(str)


def unit_histograms(fs: FeatureSet, labels: np.ndarray, n_bins: int, units: np.ndarray) -> np.ndarray:
    n_units = int(units.max()) + 1
    H = np.zeros((n_units, n_bins))
    np.add.at(H, (units[fs.clip], labels), 1.0)
    return H


def _families(V: np.ndarray, threshold: float, max_exact: int, seed: int) -> np.ndarray:
    """Average-linkage families of rows of V (sqrt histograms); threshold in Hellinger distance."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    n = len(V)
    if n == 1:
        return np.zeros(1, np.int64)
    if n > max_exact:
        from sklearn.cluster import MiniBatchKMeans

        pre = MiniBatchKMeans(max_exact, random_state=seed, batch_size=8192, n_init=1).fit_predict(V)
        cent = np.zeros((max_exact, V.shape[1]))
        np.add.at(cent, pre, V)
        cent /= np.maximum(np.bincount(pre, minlength=max_exact), 1)[:, None]
        return _families(cent, threshold, max_exact, seed)[pre]
    Z = linkage(pdist(V, "euclidean"), "average")
    return fcluster(Z, t=threshold * np.sqrt(2), criterion="distance").astype(np.int64) - 1


def cluster_heldout(fs: FeatureSet, labels: np.ndarray, n_bins: int, categories: np.ndarray, frac: float = 0.1,
                    threshold: float = 0.7, seed: int = 0, slack: float = 0.15, patterns: tuple[str, ...] | None = None,
                    max_exact: int = 25000) -> tuple[dict, np.ndarray, np.ndarray]:
    """Hold out whole motion families per category until ``frac`` of its hours (``labels``: coarse cluster per window).

    ``categories`` is per clip; a recording unit (and every clip in it) takes the category of most of its hours.
    Returns (split dict, held-out clip mask, category per clip).
    """
    rec, units, uh, ucat = _units(fs, categories, patterns)
    n_units = len(uh)
    H = unit_histograms(fs, labels, n_bins, units)
    has = H.sum(1) > 0
    V = np.sqrt(H / np.maximum(H.sum(1, keepdims=True), 1))
    fam = np.full(n_units, -1, np.int64)
    nxt = 0
    for c in np.unique(ucat):
        m = np.flatnonzero((ucat == c) & has)
        if len(m):
            f = _families(V[m], threshold, max_exact, seed)
            fam[m] = f + nxt
            nxt += int(f.max()) + 1
        for u in np.flatnonzero((ucat == c) & ~has):
            fam[u] = nxt
            nxt += 1
    fam_hours = np.bincount(fam, weights=uh, minlength=nxt)
    fam_cat = np.empty(nxt, object)
    fam_cat[fam] = ucat
    rng = np.random.default_rng(seed)
    held_fam = np.zeros(nxt, bool)
    per_cat_target = {}
    for c in np.unique(ucat):
        budget = frac * uh[ucat == c].sum()
        got = 0.0
        fams = np.unique(fam[ucat == c])
        for f in rng.permutation(fams):
            if got >= budget:
                break
            if got + fam_hours[f] <= budget * (1 + slack):
                held_fam[f] = True
                got += fam_hours[f]
        per_cat_target[str(c)] = {"budget_hours": float(budget), "heldout_hours": float(got), "families": int(len(fams)),
                                  "heldout_families": int(held_fam[fams].sum())}
    held, moved = _propagate(held_fam[fam][units], rec)
    sizes = np.bincount(fam, minlength=nxt)
    fam_summary = {"n_families": int(nxt), "recordings_per_family_median": float(np.median(sizes)),
                   "recordings_per_family_max": int(sizes.max()), "singleton_families": int((sizes == 1).sum()),
                   "largest_family_hours": float(fam_hours.max())}
    clip_fam = fam[units]
    heldout_families = []
    for f in np.flatnonzero(held_fam):
        clips = np.flatnonzero(clip_fam == f)
        heldout_families.append({"family": int(f), "category": str(fam_cat[f]), "recordings": int(sizes[f]),
                                 "clips": int(len(clips)), "hours": float(fam_hours[f]),
                                 "groups": dict(Counter(fs.clip_group[clips].astype(str))),
                                 "examples": [str(fs.clip_name[i]) for i in clips[:3]]})
    params = {"frac": frac, "threshold_hellinger": threshold, "n_bins": n_bins, "seed": seed, "slack": slack,
              "recording_patterns": list(patterns) if patterns is not None else "default"}
    out = _finish("cluster_heldout", fs, held, rec, ucat[units], params,
                  {"families": fam_summary, "category_budget": per_cat_target, "added_by_shared_recording": moved})
    out["heldout_families"] = heldout_families
    return out, held, ucat[units]


def _units(fs: FeatureSet, categories: np.ndarray, patterns: tuple[str, ...] | None):
    """Recording id per clip, unit index per clip, hours per unit, category per unit (majority by hours)."""
    rec = recordings(fs, patterns)
    uniq, units = np.unique(rec, return_inverse=True)
    h = fs.clip_hours()
    uh = np.bincount(units, weights=h, minlength=len(uniq))
    cats, ci = np.unique(np.asarray(categories).astype(str), return_inverse=True)
    M = np.zeros((len(uniq), len(cats)))
    np.add.at(M, (units, ci), h + 1e-9)
    return rec, units, uh, cats[M.argmax(1)]


def random_unit_split(fs: FeatureSet, categories: np.ndarray, frac_by_cat: dict[str, float], seed: int = 0,
                      patterns: tuple[str, ...] | None = None) -> np.ndarray:
    """Baseline: random recording units per category until the same held-out fractions (held-out clip mask)."""
    _, units, uh, ucat = _units(fs, categories, patterns)
    rng = np.random.default_rng(seed)
    held_u = np.zeros(len(uh), bool)
    for c, f in frac_by_cat.items():
        idx = np.flatnonzero(ucat == c)
        budget, got = f * uh[idx].sum(), 0.0
        for u in rng.permutation(idx):
            if got >= budget:
                break
            held_u[u] = True
            got += uh[u]
    return held_u[units]


def split_novelty(fs: FeatureSet, P: np.ndarray, held: np.ndarray, categories: np.ndarray, baseline: np.ndarray | None = None,
                  max_query: int = 200000, seed: int = 0, device: str | None = None) -> dict:
    """Median 1-NN distance of held-out windows to the training-side windows (per category), vs. a baseline split."""
    rng = np.random.default_rng(seed)

    def one(mask):
        wh = mask[fs.clip]
        q = np.flatnonzero(wh)
        if len(q) > max_query:
            q = np.sort(rng.choice(q, max_query, replace=False))
        d = knn(P[q], P[~wh], 1, device=device)[0][:, 0]
        wc = categories[fs.clip[q]]
        res = {str(c): float(np.median(d[wc == c])) for c in np.unique(wc)}
        res["all"] = float(np.median(d))
        return res

    out = {"heldout_d1_median": one(held)}
    if baseline is not None:
        out["random_unit_split_d1_median"] = one(baseline)
    return out


def add_novelty(split: dict, fs: FeatureSet, P: np.ndarray, held: np.ndarray, categories: np.ndarray, seed: int = 0,
                device: str | None = None) -> dict:
    """Attach ``split_novelty`` against a random unit split with the same per-category held-out fractions."""
    frac = {c: v["heldout_frac"] for c, v in split["summary"]["per_category"].items()}
    base = random_unit_split(fs, categories, frac, seed)
    split["summary"]["novelty"] = split_novelty(fs, P, held, categories, base, seed=seed, device=device)
    return split
