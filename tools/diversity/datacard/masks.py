"""Per-window feasibility masks used to restrict data-card metrics.

Mask file format (npz):

* ``clip_path``: str array (N,), motion file of the window's clip; matched on the resolved absolute path
  (``os.path.realpath``), so the mask must reference the same files as the feature set.
* ``start``: int array (N,), first frame of the window in that clip.
* ``feasible``: bool array (N,), whether the window passes the feasibility checks.
* ``window_frames`` (optional): int scalar or int array (N,), window length in frames. Defaults to the data-card
  window (1 s, i.e. 50 frames at 50 Hz). Clip-level masks can use ``start = 0`` and ``window_frames = clip length``.

Each data-card window ``[start, start + W)`` of a clip is resolved as follows:

1. an entry with the same clip, start and length gives the flag directly (use the data-card grid: ``W = round(1 s /
   dt)``, starts every ``round(0.5 s / dt)`` frames, to get exact matches);
2. otherwise the entries of that clip overlapping the window decide: feasible only if all of them are feasible;
3. otherwise (clip not in the mask, or no overlapping entry) the ``missing`` policy applies: ``"keep"`` (default,
   counted as feasible) or ``"drop"``.
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from .features import FeatureSet


def _key(path: str) -> str:
    return os.path.realpath(os.path.abspath(path))


def load_mask(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        m = {"clip_path": z["clip_path"].astype(str), "start": z["start"].astype(np.int64),
             "feasible": z["feasible"].astype(bool)}
        if "window_frames" in z:
            m["window_frames"] = z["window_frames"]
    n = len(m["clip_path"])
    if len(m["start"]) != n or len(m["feasible"]) != n:
        raise ValueError(f"{path}: clip_path / start / feasible lengths differ")
    return m


def window_mask(fs: FeatureSet, mask: dict | str | Path, missing: str = "keep") -> tuple[np.ndarray, dict]:
    """Feasible flag per window of ``fs`` and match statistics."""
    if missing not in ("keep", "drop"):
        raise ValueError("missing must be 'keep' or 'drop'")
    if not isinstance(mask, dict):
        mask = load_mask(mask)
    W_default = fs.window_frames
    wf = mask.get("window_frames")
    n = len(mask["clip_path"])
    lens = np.full(n, -1, np.int64) if wf is None else np.broadcast_to(np.asarray(wf, np.int64), (n,)).copy()
    by_clip = defaultdict(list)
    for i, p in enumerate(mask["clip_path"]):
        by_clip[_key(p)].append(i)
    out = np.ones(fs.n_windows, bool)
    status = np.zeros(fs.n_windows, np.int8)  # 0 missing, 1 exact, 2 overlap
    order = np.argsort(fs.clip, kind="stable")
    bounds = np.searchsorted(fs.clip[order], np.arange(fs.n_clips + 1))
    for c in range(fs.n_clips):
        w = order[bounds[c]:bounds[c + 1]]
        idx = by_clip.get(_key(str(fs.clip_path[c])))
        if not idx or len(w) == 0:
            continue
        idx = np.array(idx)
        W = int(W_default[c])
        e_start, e_feas = mask["start"][idx], mask["feasible"][idx]
        e_len = np.where(lens[idx] < 0, W, lens[idx])
        s = fs.start[w].astype(np.int64)
        exact = {int(a): bool(f) for a, f, ln in zip(e_start, e_feas, e_len) if ln == W}
        for wi, si in zip(w, s):
            if int(si) in exact:
                out[wi], status[wi] = exact[int(si)], 1
                continue
            ov = (e_start < si + W) & (e_start + e_len > si)
            if ov.any():
                out[wi], status[wi] = bool(e_feas[ov].all()), 2
    if missing == "drop":
        out[status == 0] = False
    stats = {"windows": int(fs.n_windows), "exact": int((status == 1).sum()), "overlap": int((status == 2).sum()),
             "missing": int((status == 0).sum()), "missing_policy": missing, "feasible": int(out.sum()),
             "feasible_frac": float(out.mean()) if fs.n_windows else float("nan")}
    return out, stats
