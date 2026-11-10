"""Target-centred coverage and novelty of a dataset with respect to a target set, in the frozen embedding.

* Novelty of a target window = distance to the nearest dataset window (1-NN; also the k-th NN and the mean of the
  k NN distances). Per target clip: mean and p90 over its windows. The clip mean of the 1-NN distance is the primary
  per-clip novelty (it tracks per-clip body-position error beyond difficulty; it does not predict failures).
* Naeem et al. (2020) coverage, oriented on the target: every target window gets a ball whose radius is the distance
  to its k-th nearest target window from another clip (k = 5); the window is covered if any dataset window falls in
  it. It can only grow when data is added, so it is comparable across dataset versions. (Balls centred on the
  training windows are not monotone: adding data shrinks the radii.)
* Fixed-radius coverage: share of target windows with a dataset window within one radius tau.
* Leave-one-group-out: coverage with each dataset group removed, from per-group 1-NN distances (one pass).
"""

from __future__ import annotations

import numpy as np

from .knn import knn


def target_radii(P_target: np.ndarray, clip: np.ndarray, k: int = 5, device: str | None = None) -> np.ndarray:
    """Distance from each target window to its k-th nearest target window of a different clip."""
    d, _ = knn(P_target, P_target, k, clip, clip, device=device)
    return d[:, k - 1]


def nn_distances(P_target: np.ndarray, P_data: np.ndarray, k: int = 10, device: str | None = None) -> np.ndarray:
    return knn(P_target, P_data, k, device=device)[0]


def group_nn(P_target: np.ndarray, P_data: np.ndarray, data_groups: np.ndarray, device: str | None = None
             ) -> dict[str, np.ndarray]:
    """1-NN distance from each target window to each group of the dataset."""
    return {str(g): knn(P_target, P_data[data_groups == g], 1, device=device)[0][:, 0] for g in np.unique(data_groups)}


def by_category(values: np.ndarray, cat: np.ndarray, cats=None) -> dict:
    cats = [str(c) for c in (np.unique(cat) if cats is None else cats)]
    out = {c: float(np.mean(values[cat == c])) if np.any(cat == c) else float("nan") for c in cats}
    out["all"] = float(np.mean(values))
    return out


def coverage(d1: np.ndarray, radii: np.ndarray | None = None, tau: float | None = None) -> np.ndarray:
    """Covered target windows: d1 <= own ball radius (Naeem) or d1 <= tau (fixed radius)."""
    if (radii is None) == (tau is None):
        raise ValueError("give exactly one of radii / tau")
    return d1 <= (radii if radii is not None else tau)


def leave_one_group_out(d1_by_group: dict[str, np.ndarray], radii: np.ndarray, cat: np.ndarray,
                        extra: dict[str, list[str]] | None = None) -> dict:
    """Naeem coverage per category with all groups, and with each group (or named group set in ``extra``) removed."""
    groups = list(d1_by_group)
    D = np.stack([d1_by_group[g] for g in groups], 1)
    out = {"all": by_category(D.min(1) <= radii, cat)}
    sets = {g: [g] for g in groups}
    sets.update(extra or {})
    for name, drop in sets.items():
        keep = [i for i, g in enumerate(groups) if g not in drop]
        d1 = D[:, keep].min(1) if keep else np.full(len(D), np.inf)
        out[f"-{name}"] = by_category(d1 <= radii, cat)
    return out


def per_clip(values: np.ndarray, clip: np.ndarray, n_clips: int, fn=np.mean) -> np.ndarray:
    order = np.argsort(clip, kind="stable")
    bounds = np.searchsorted(clip[order], np.arange(n_clips + 1))
    return np.array([fn(values[order[a:b]]) if b > a else np.nan for a, b in zip(bounds[:-1], bounds[1:])])


def clip_novelty(d: np.ndarray, clip: np.ndarray, n_clips: int, radii: np.ndarray | None = None,
                 tau: float | None = None) -> dict[str, np.ndarray]:
    """Per-clip novelty from the (n_windows, k) NN distance matrix ``d`` of the target windows."""
    p90 = lambda x: np.percentile(x, 90)  # noqa: E731
    k = d.shape[1]
    out = {"d1_mean": per_clip(d[:, 0], clip, n_clips), "d1_p90": per_clip(d[:, 0], clip, n_clips, p90),
           f"d{k}_mean": per_clip(d[:, -1], clip, n_clips), f"d{k}_p90": per_clip(d[:, -1], clip, n_clips, p90),
           f"knn{k}_mean": per_clip(d.mean(1), clip, n_clips)}
    if radii is not None:
        out["naeem_cov"] = per_clip((d[:, 0] <= radii).astype(float), clip, n_clips)
    if tau is not None:
        out["tau_cov"] = per_clip((d[:, 0] <= tau).astype(float), clip, n_clips)
    return out
