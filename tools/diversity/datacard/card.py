"""Data card: what a dataset version covers relative to a target (evaluation) set, under a frozen reference.

The reference (``Reference``: embedding + codebook + fixed radius tau) is fitted once on a declared pool and reused
for every dataset version, so the numbers of different cards are comparable. A card reports (JSON + markdown + CSV):

* size: hours, clips, sources, recordings, per-group hours / clips / sampling weight;
* codebook occupancy: clusters occupied (>= 1 / >= 5 windows), hours per cluster, effective number of clusters,
  Gini, top-1% / top-10% share, also under the training-mixture weights; target-window occupancy per category;
* coverage per target category: Naeem coverage with target-centred k-NN balls (k = 5; monotone in the data, the
  version-comparison metric), fixed-radius coverage at tau, median 1-NN distance; leave-one-group-out coverage;
* per-target-clip novelty table (1-NN / k-NN distances, mean and p90, coverage shares, difficulty), least-covered
  target clips, and segments duplicated between the dataset and the target set (``dedup``).

Feasibility masks (optional) restrict the dataset and / or target windows used by every metric. Mask file (npz):
``clip_path`` (str array, motion file of the window's clip, matched on the resolved absolute path), ``start`` (int
array, first frame of the window), ``feasible`` (bool array), optional ``window_frames`` (int scalar or array; default
the 1 s data-card window). Exact (clip, start) matches on the data-card grid (``W = round(1 s / dt)``, stride
``round(0.5 s / dt)``) are used directly; other windows are feasible only if all overlapping mask entries of their clip
are; windows of clips absent from the mask follow ``missing`` ("keep" or "drop"). See ``masks.py``.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .codebook import Codebook, fit_codebook, occupancy, target_occupancy
from .coverage import by_category, clip_novelty, group_nn, leave_one_group_out, nn_distances, target_radii
from .datasets import recording_id
from .embed import Embedding, fit_embedding
from .features import DIFFICULTY_KEYS, FeatureSet
from .knn import knn
from .masks import window_mask

CARD_VERSION = 1


@dataclass
class Reference:
    embedding: Embedding
    codebook: Codebook
    tau: float
    meta: dict = field(default_factory=dict)

    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.embedding.save(out / "embedding.npz")
        self.codebook.save(out / "codebook.npz")
        with open(out / "reference.json", "w") as f:
            json.dump(dict(self.meta, tau=self.tau), f, indent=1)

    @classmethod
    def load(cls, out_dir: str | Path) -> "Reference":
        out = Path(out_dir)
        emb, cb = Embedding.load(out / "embedding.npz"), Codebook.load(out / "codebook.npz")
        with open(out / "reference.json") as f:
            meta = json.load(f)
        if cb.meta.get("embedding") not in (None, emb.fingerprint):
            raise ValueError(f"{out}: codebook was fitted on another embedding")
        return cls(emb, cb, float(meta["tau"]), meta)


def fit_reference(pool: FeatureSet, description: str, n_components: int = 32, k: int = 2000, seed: int = 0,
                  tau_k: int = 5, tau_sample: int = 50000, device: str | None = None, verbose: bool = True) -> Reference:
    """Embedding + codebook on the pool; tau = median distance of pool windows to their ``tau_k``-th nearest pool
    window from another source (computed on a sample)."""
    t0 = time.time()
    emb = fit_embedding(pool, n_components, seed=seed, description=description)
    P = emb.transform(pool.X)
    cb = fit_codebook(P, k, seed, device=device, description=description, embedding=emb.fingerprint, verbose=verbose)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(P), min(tau_sample, len(P)), replace=False))
    _, src = np.unique(pool.clip_source[pool.clip], return_inverse=True)
    tau = float(np.median(knn(P[idx], P, tau_k, src[idx], src, device=device)[0][:, -1]))
    meta = {"description": description, "embedding_fingerprint": emb.fingerprint, "robot": pool.robot,
            "explained_variance": emb.meta["explained_variance"], "k": k, "tau_definition":
            f"median {tau_k}-NN distance between pool windows of different sources ({len(idx)} sampled windows)",
            "pool": emb.meta["pool"], "seconds": round(time.time() - t0, 1)}
    if verbose:
        print(f"[reference] PCA{n_components} explained {emb.meta['explained_variance']:.3f}, k={k}, tau {tau:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    return Reference(emb, cb, tau, meta)


def mixture_weights(fs: FeatureSet, weights: dict | None = None, subsets: dict | None = None) -> tuple[np.ndarray | None, list]:
    """Window sampling probabilities of a training mixture: each group / subset gets its weight, spread uniformly over
    its windows (subsets such as re-weighted ``getup`` clips add to their groups). Returns (p or None, missing keys)."""
    weights = fs.meta.get("weights") if weights is None else weights
    subsets = fs.meta.get("subsets") if subsets is None else subsets
    if not weights:
        return None, []
    p, missing = np.zeros(fs.n_windows), []
    wg, wn = fs.window_group, fs.clip_name[fs.clip]
    for g, w in weights.items():
        m = np.isin(wn, subsets[g]) if subsets and g in subsets else wg == g
        if not m.any():
            missing.append(g)
            continue
        p[m] += w / m.sum()
    return (p / p.sum() if p.sum() > 0 else None), missing


def _size(fs: FeatureSet, weights: dict | None) -> dict:
    h, rec = fs.clip_hours(), np.array([recording_id(str(s)) for s in fs.clip_source])
    nw = fs.clip_windows
    groups = {}
    for g in np.unique(fs.clip_group):
        m = fs.clip_group == g
        groups[str(g)] = {"hours": float(h[m].sum()), "clips": int(m.sum()), "windows": int(nw[m].sum()),
                          "sources": int(len(set(fs.clip_source[m]))), "recordings": int(len(set(rec[m])))}
        if weights and str(g) in weights:
            groups[str(g)]["weight"] = float(weights[str(g)])
    return {"name": fs.name, "description": fs.meta.get("description", ""), "robot": fs.robot, "hours": fs.hours(),
            "clips": fs.n_clips, "windows": fs.n_windows, "sources": int(len(set(fs.clip_source))),
            "recordings": int(len(set(rec))), "groups": groups}


def build_card(dataset: FeatureSet, target: FeatureSet, ref: Reference, k_radius: int = 5, k_novelty: int = 10,
               tau: float | None = None, dataset_mask=None, target_mask=None, mask_missing: str = "keep",
               dedup: dict | None = None, logo_extra: dict | None = None, n_least: int = 10,
               device: str | None = None) -> tuple[dict, dict]:
    """Returns (card dict, arrays for the npz sidecar)."""
    t0 = time.time()
    emb, cb = ref.embedding, ref.codebook
    tau = ref.tau if tau is None else float(tau)
    Pd, Pt = emb.embed(dataset), emb.embed(target)
    dm, dstats = window_mask(dataset, dataset_mask, mask_missing) if dataset_mask is not None else (np.ones(len(Pd), bool), None)
    tm, tstats = window_mask(target, target_mask, mask_missing) if target_mask is not None else (np.ones(len(Pt), bool), None)
    tclip, tcat = target.clip[tm], target.window_group[tm].astype(str)
    cats = [str(c) for c in np.unique(target.clip_group)]
    Pt_m, Pd_m = Pt[tm], Pd[dm]
    radii = target_radii(Pt_m, tclip, k_radius, device)
    d = nn_distances(Pt_m, Pd_m, k_novelty, device)
    d1 = d[:, 0]
    naeem, fixed = d1 <= radii, d1 <= tau
    nov = clip_novelty(d, tclip, target.n_clips, radii, tau)
    cov = {"k_radius": k_radius, "tau": tau, "tau_definition": ref.meta.get("tau_definition", "user") if tau == ref.tau
           else "user", "naeem": by_category(naeem, tcat, cats), "fixed_radius": by_category(fixed, tcat, cats),
           "d1_median": {c: float(np.median(d1[tcat == c])) for c in cats} | {"all": float(np.median(d1))},
           "target_radius_median": {c: float(np.median(radii[tcat == c])) for c in cats} | {"all": float(np.median(radii))},
           "clip_mean_naeem": {c: float(np.nanmean(nov["naeem_cov"][target.clip_group == c])) for c in cats}}
    if dataset_mask is not None:
        d1_all = nn_distances(Pt_m, Pd, 1, device)[:, 0]
        cov["naeem_unmasked"] = by_category(d1_all <= radii, tcat, cats)
    groups = dataset.window_group[dm].astype(str)
    logo = None
    if len(np.unique(groups)) > 1:
        logo = leave_one_group_out(group_nn(Pt_m, Pd_m, groups, device), radii, tcat, logo_extra)

    lab_d, lab_t = cb.assign(Pd_m, device), cb.assign(Pt_m, device)
    w, missing_w = mixture_weights(dataset)
    occ, cluster_hours = occupancy(lab_d, cb.k, None if w is None else w[dm] / max(w[dm].sum(), 1e-12),
                                   stride_s=dataset.meta["stride_s"])
    if missing_w:
        occ["weights_missing_groups"] = missing_w
    tocc = target_occupancy(lab_d, lab_t, tcat, cb.k)

    dup_clips = (dedup or {}).get("b_clips", {})
    tw = np.bincount(tclip, minlength=target.n_clips)
    rows = []
    for i in range(target.n_clips):
        r = {"clip": str(target.clip_name[i]), "category": str(target.clip_group[i]),
             "windows": int(tw[i]), **{k: _f(v[i]) for k, v in nov.items()},
             **{k: _f(target.diff[i, j]) for j, k in enumerate(DIFFICULTY_KEYS)}}
        if dedup is not None:
            dc = dup_clips.get(r["clip"], {})
            r["duplicated"] = dc.get("whole_a_matches", 0) > 0
            r["partial_overlap"] = bool(dc) and not r["duplicated"]
            r["duplicated_frac"] = dc.get("matched_frac", 0.0)
        rows.append(r)
    ranked = sorted([r for r in rows if r["d1_mean"] is not None], key=lambda r: -r["d1_mean"])
    least = {"overall": ranked[:n_least],
             "per_category": {c: [r for r in ranked if r["category"] == c][:3] for c in cats},
             "lowest_naeem": sorted([r for r in rows if r.get("naeem_cov") is not None],
                                    key=lambda r: (r["naeem_cov"], -r["d1_mean"]))[:n_least]}
    card = {"card_version": CARD_VERSION, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": _size(dataset, dataset.meta.get("weights")), "target": _size(target, None),
            "reference": {"description": ref.meta.get("description"), "embedding": emb.fingerprint,
                          "explained_variance": emb.meta.get("explained_variance"), "codebook_k": cb.k,
                          "pool_hours": emb.meta.get("pool", {}).get("hours")},
            "masks": {"dataset": dstats, "target": tstats},
            "occupancy": occ, "target_occupancy": tocc, "coverage": cov, "leave_one_group_out": logo,
            "novelty_clips": rows, "least_covered": least, "duplicates": _dup_summary(dedup, target, cats),
            "seconds": round(time.time() - t0, 1)}
    arrays = {"cluster_hours": cluster_hours, "target_d1": d1, "target_radius": radii, "target_clip": tclip,
              "target_start": target.start[tm], "dataset_cluster": lab_d, "target_cluster": lab_t}
    return card, arrays


def _f(x):
    return None if x is None or np.isnan(x) else float(x)


def _dup_summary(dedup: dict | None, target: FeatureSet, cats: list[str]) -> dict | None:
    if dedup is None:
        return None
    b = dedup["b_clips"]
    whole = {n: v for n, v in b.items() if v.get("whole_a_matches", 0) > 0}
    return {"params": dedup["params"], "n_target_clips_whole_segment": len(whole), "n_target_clips_any": len(b),
            "per_category": {c: sum(v["group"] == c for v in whole.values()) for c in cats},
            "per_category_any": {c: sum(v["group"] == c for v in b.values()) for c in cats},
            "per_category_total": {c: int(np.sum(target.clip_group == c)) for c in cats},
            "target_clips": b, "n_matches": len(dedup["matches"])}


def card_markdown(card: dict) -> str:
    ds, tg, cov, occ = card["dataset"], card["target"], card["coverage"], card["occupancy"]
    cats = [c for c in cov["naeem"] if c != "all"]
    tg_groups = ", ".join(f"{c} {v['clips']}" for c, v in tg["groups"].items())
    L = [f"# Data card: {ds['name']} vs {tg['name']}", ""]
    if ds.get("description"):
        L += [ds["description"], ""]
    L += [f"- Robot `{ds['robot']}`; reference: {card['reference']['description']} (embedding "
          f"`{card['reference']['embedding']}`, PCA explained {card['reference']['explained_variance']:.3f}, "
          f"codebook k={card['reference']['codebook_k']})",
          f"- Dataset: {ds['hours']:.2f} h, {ds['clips']} clips, {ds['sources']} sources, {ds['recordings']} recordings, "
          f"{ds['windows']} windows",
          f"- Target: {tg['hours']:.2f} h, {tg['clips']} clips ({tg_groups})"]
    for side in ("dataset", "target"):
        m = card["masks"][side]
        if m:
            L.append(f"- Feasibility mask ({side}): {m['feasible']}/{m['windows']} windows feasible "
                     f"({m['feasible_frac']:.3f}); exact {m['exact']}, overlap {m['overlap']}, missing {m['missing']} "
                     f"({m['missing_policy']})")
    L += ["", "## Groups", "", "| group | hours | clips | sources | weight |", "|---|---:|---:|---:|---:|"]
    for g, v in ds["groups"].items():
        L.append(f"| {g} | {v['hours']:.2f} | {v['clips']} | {v['sources']} | {v.get('weight', '')} |")
    L += ["", "## Codebook occupancy", "",
          f"- Occupied clusters: {occ['occupied_frac']['1']:.3f} (>= 1 window), {occ['occupied_frac']['5']:.3f} (>= 5)",
          f"- Effective clusters {occ['effective_clusters']:.0f} of {occ['k']}, Gini {occ['gini']:.3f}, "
          f"top-1% share {occ['top1pct_share']:.3f}, top-10% share {occ['top10pct_share']:.3f}"]
    if "weighted" in occ:
        w = occ["weighted"]
        L.append(f"- Under mixture weights: effective clusters {w['effective_clusters']:.0f}, Gini {w['gini']:.3f}, "
                 f"top-10% share {w['top10pct_share']:.3f}")
    L += ["", "## Coverage of the target", "",
          f"Naeem coverage: target window covered if a dataset window lies within its {cov['k_radius']}-NN target ball; "
          f"fixed radius tau = {cov['tau']:.3f}; occupancy = target windows in clusters with >= 1 dataset window.", "",
          "| category | Naeem | fixed tau | occupancy | median d1 |", "|---|---:|---:|---:|---:|"]
    for c in cats + ["all"]:
        L.append(f"| {c} | {cov['naeem'][c]:.3f} | {cov['fixed_radius'][c]:.3f} | "
                 f"{card['target_occupancy'][c]['window_frac']['1']:.3f} | {cov['d1_median'][c]:.2f} |")
    if "naeem_unmasked" in cov:
        L.append("\nNaeem without the dataset mask: " + ", ".join(f"{c} {v:.3f}" for c, v in cov["naeem_unmasked"].items()))
    logo = card.get("leave_one_group_out")
    if logo:
        L += ["", "## Leave-one-group-out (Naeem)", "", "| without | " + " | ".join(cats) + " |",
              "|---|" + "---:|" * len(cats)]
        for k, v in logo.items():
            L.append(f"| {k} | " + " | ".join(f"{v[c]:.3f}" for c in cats) + " |")
    L += ["", "## Least-covered target clips (clip mean 1-NN distance)", "",
          "| clip | category | d1 mean | d1 p90 | Naeem | dur s |", "|---|---|---:|---:|---:|---:|"]
    for r in card["least_covered"]["overall"]:
        L.append(f"| {r['clip'][-60:]} | {r['category']} | {r['d1_mean']:.2f} | {r['d1_p90']:.2f} | "
                 f"{r.get('naeem_cov', float('nan')):.2f} | {r['dur_s']:.1f} |")
    dup = card.get("duplicates")
    if dup:
        p = dup["params"]
        L += ["", "## Target clips duplicated in the dataset", "",
              f"{dup['n_target_clips_whole_segment']} target clips contain a whole dataset clip (>= {p['min_len_s']} s) "
              f"with {p['metric']} |dq| <= {p['tol']} rad over it. Per category: " + ", ".join(
                  f"{c} {dup['per_category'][c]}/{dup['per_category_total'][c]}" for c in cats) +
              f". {dup['n_target_clips_any'] - dup['n_target_clips_whole_segment']} more share only a partial overlap "
              f"(>= {p.get('partial_min_len_s')} s, joint range >= {p.get('partial_min_motion')} rad).", ""]
        items = sorted(dup["target_clips"].items(), key=lambda kv: (kv[1]["whole_a_matches"] == 0, kv[0]))
        for n, v in items:
            kind = (f"{v['whole_a_matches']} whole dataset clips, e.g. `{v['whole_a_clips'][0]}`"
                    if v["whole_a_matches"] else f"partial overlap with `{v['a_clips'][0]}`")
            L.append(f"- `{n}` ({v['group']}): {kind}; {v['matched_frac']:.2f} of the clip matched")
    return "\n".join(L) + "\n"


def write_card(card: dict, arrays: dict, out_dir: str | Path, stem: str) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"json": out / f"{stem}.json", "md": out / f"{stem}.md", "csv": out / f"{stem}_novelty.csv",
             "npz": out / f"{stem}_arrays.npz"}
    with open(paths["json"], "w") as f:
        json.dump(card, f, indent=1)
    paths["md"].write_text(card_markdown(card))
    rows = card["novelty_clips"]
    with open(paths["csv"], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    np.savez(paths["npz"], **arrays)
    return {k: str(v) for k, v in paths.items()}
