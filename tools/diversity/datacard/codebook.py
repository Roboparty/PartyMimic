"""k-means codebook over the frozen window embedding and occupancy metrics.

The codebook (default 2000 clusters) is fitted on the embedded reference pool and frozen together with the
embedding (``meta["embedding"]`` stores the embedding fingerprint). Occupancy of a dataset = how its windows spread
over the clusters: fraction of clusters occupied (>= m windows), hours per cluster, effective number of clusters
(exp of the entropy of the cluster distribution), Gini and top-k share; optionally under the sampling weights of the
training mixture. Target occupancy = share of target windows (per category) that fall into clusters the dataset
occupies.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .knn import default_device, nearest_center, torch

MIN_WINDOWS = (1, 5)


@dataclass
class Codebook:
    centers: np.ndarray
    ref_counts: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def k(self) -> int:
        return len(self.centers)

    def assign(self, P: np.ndarray, device: str | None = None) -> np.ndarray:
        return nearest_center(P, self.centers, device)[0]

    def coarse_map(self, n: int = 128, seed: int = 0) -> np.ndarray:
        """Fine -> coarse cluster map (weighted k-means of the centers, weights = reference occupancy)."""
        from sklearn.cluster import KMeans

        n = min(n, self.k)
        km = KMeans(n, random_state=seed, n_init=4).fit(self.centers, sample_weight=np.maximum(self.ref_counts, 1))
        return km.labels_.astype(np.int64)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, centers=self.centers, ref_counts=self.ref_counts, meta_json=np.array(json.dumps(self.meta)))

    @classmethod
    def load(cls, path: str | Path) -> "Codebook":
        with np.load(path, allow_pickle=False) as z:
            return cls(z["centers"], z["ref_counts"], json.loads(str(z["meta_json"])))


def _kmeanspp(X: np.ndarray, k: int, seed: int, device: str) -> np.ndarray:
    gen = np.random.default_rng(seed)
    n = len(X)
    idx = [int(gen.integers(n))]
    if device != "numpy" and torch is not None:
        Xt = torch.from_numpy(X).to(device)
        d2 = ((Xt - Xt[idx[0]]) ** 2).sum(1)
        for _ in range(1, k):
            p = d2.double().cpu().numpy()
            j = int(gen.choice(n, p=p / p.sum()))
            idx.append(j)
            d2 = torch.minimum(d2, ((Xt - Xt[j]) ** 2).sum(1))
    else:
        d2 = ((X - X[idx[0]]) ** 2).sum(1)
        for _ in range(1, k):
            p = d2.astype(np.float64)
            j = int(gen.choice(n, p=p / p.sum()))
            idx.append(j)
            d2 = np.minimum(d2, ((X - X[j]) ** 2).sum(1))
    return X[idx].copy()


def fit_codebook(P: np.ndarray, k: int = 2000, seed: int = 0, iters: int = 100, tol: float = 1e-4,
                 device: str | None = None, init_sample: int = 200000, method: str = "auto", description: str = "",
                 embedding: str | None = None, verbose: bool = True) -> Codebook:
    """Lloyd k-means (k-means++ init on a subsample, GPU assignment, deterministic CPU update) or MiniBatchKMeans."""
    t0 = time.time()
    dev = default_device(device)
    P = np.ascontiguousarray(P, np.float32)
    if len(P) < k:
        raise ValueError(f"{len(P)} windows < {k} clusters")
    if method == "minibatch" or (method == "auto" and (dev == "numpy" or torch is None) and len(P) > 200000):
        from sklearn.cluster import MiniBatchKMeans

        km = MiniBatchKMeans(k, random_state=seed, batch_size=8192, n_init=1, max_iter=iters).fit(P)
        C, it, method = km.cluster_centers_.astype(np.float32), int(km.n_iter_), "minibatch"
    else:
        rng = np.random.default_rng(seed)
        sub = P[np.sort(rng.choice(len(P), min(init_sample, len(P)), replace=False))]
        C = _kmeanspp(sub, k, seed, dev)
        prev = np.inf
        for it in range(1, iters + 1):
            lab, d2 = nearest_center(P, C, dev)
            inertia = float(d2.astype(np.float64).sum())
            counts = np.bincount(lab, minlength=k)
            sums = np.stack([np.bincount(lab, weights=P[:, j], minlength=k) for j in range(P.shape[1])], 1)
            C = (sums / np.maximum(counts, 1)[:, None]).astype(np.float32)
            empty = np.flatnonzero(counts == 0)
            if len(empty):
                C[empty] = P[np.argsort(-d2)[:len(empty)]]
            if verbose and (it % 10 == 0 or it == 1):
                print(f"[codebook] iter {it} inertia {inertia:.4g} empty {len(empty)} ({time.time() - t0:.0f}s)", flush=True)
            if len(empty) == 0 and prev - inertia <= tol * inertia:
                break
            prev = inertia
        method = "lloyd"
    lab, d2 = nearest_center(P, C, dev)
    counts = np.bincount(lab, minlength=k)
    meta = {"k": k, "seed": seed, "method": method, "iters": it, "n_fit": len(P), "dim": int(P.shape[1]),
            "inertia_per_window": float(d2.astype(np.float64).mean()), "embedding": embedding,
            "pool_description": description, "empty_on_pool": int((counts == 0).sum()),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"), "fit_seconds": round(time.time() - t0, 1)}
    if verbose:
        print(f"[codebook] k={k} {method} iters {it} empty {meta['empty_on_pool']} ({time.time() - t0:.0f}s)", flush=True)
    return Codebook(C, counts, meta)


def _spread(mass: np.ndarray, top_fracs=(0.01, 0.1)) -> dict:
    k = len(mass)
    nz = mass[mass > 0]
    asc = np.sort(mass)
    gini = float((2 * np.sum(np.arange(1, k + 1) * asc) / (k * asc.sum())) - (k + 1) / k) if asc.sum() > 0 else 0.0
    desc = asc[::-1]
    return {"effective_clusters": float(np.exp(-(nz * np.log(nz)).sum())), "gini": gini,
            **{f"top{f * 100:g}pct_share": float(desc[:max(1, int(np.ceil(f * k)))].sum()) for f in top_fracs}}


def occupancy(labels: np.ndarray, k: int, weights: np.ndarray | None = None, min_windows=MIN_WINDOWS,
              stride_s: float = 0.5) -> tuple[dict, np.ndarray]:
    """Occupancy of a dataset's windows (``labels`` = cluster per window); ``weights`` = window sampling weights.

    Returns (summary dict, hours per cluster (k,)).
    """
    counts = np.bincount(labels, minlength=k)
    hours = counts * stride_s / 3600.0
    occ = hours[counts > 0]
    out = {"k": k, "windows": int(counts.sum()),
           "occupied_frac": {str(m): float(np.mean(counts >= m)) for m in min_windows},
           "occupied_clusters": {str(m): int(np.sum(counts >= m)) for m in min_windows},
           "cluster_hours": {"p10": float(np.percentile(occ, 10)), "median": float(np.median(occ)),
                             "p90": float(np.percentile(occ, 90)), "max": float(occ.max())} if len(occ) else {},
           **_spread(counts / max(counts.sum(), 1))}
    if weights is not None:
        wm = np.bincount(labels, weights=weights, minlength=k)
        out["weighted"] = _spread(wm / wm.sum())
    return out, hours


def target_occupancy(data_labels: np.ndarray, target_labels: np.ndarray, target_cat: np.ndarray, k: int,
                     min_windows=MIN_WINDOWS) -> dict:
    """Per target category: share of target windows (and of target clusters) in clusters the dataset occupies."""
    counts = np.bincount(data_labels, minlength=k)
    out = {}
    cats = [str(c) for c in np.unique(target_cat)] + ["all"]
    for c in cats:
        tl = target_labels if c == "all" else target_labels[target_cat == c]
        tc = np.unique(tl)
        out[c] = {"target_windows": int(len(tl)), "target_clusters": int(len(tc)),
                  "window_frac": {str(m): float(np.mean(counts[tl] >= m)) for m in min_windows},
                  "cluster_frac": {str(m): float(np.mean(counts[tc] >= m)) for m in min_windows}}
    return out
