"""Duplicate / near-duplicate segments between two datasets in raw joint space.

Per frame ``t`` at a fixed offset ``delta`` between clip ``a`` and clip ``b``, ``e_t`` is the mean (``metric="mean"``,
default) or the max (``metric="max"``) over joints of ``|qa[t] - qb[t + delta]|``.

* Whole-segment match (the prototype rule): the shorter clip (>= ``min_len_s``) lies entirely inside the other one and
  the mean (or max) of ``e_t`` over it is ``<= tol`` (0.05 rad). Flagged ``whole_a`` / ``whole_b``; these are the
  duplicates counted in data cards.
* Partial overlap: a run of ``>= partial_min_len_s`` covered by ``min_len_s`` windows whose mean ``e_t`` is ``<= tol``
  (``metric="max"``: every frame ``e_t <= tol``), in which some joint moves by ``>= partial_min_motion`` rad. Short or
  slow runs are dropped because similar sitting / lying poses of unrelated takes pass the mean rule.

The mean rule catches the same source motion retargeted twice (small mean error, isolated per-joint spikes); the max
rule only catches copies of the same retarget.

Candidates come from windows of ``win`` frames summarised by ``blocks`` block means of the raw joint angles, every
``stride`` frames in the larger dataset and at every frame in the smaller one (``win + stride - 1 <= min_len``, so
every qualifying segment contains a grid window whose exactly aligned partner exists). Pairs within L2 radius
``cand_factor * tol * sqrt(blocks * nj)`` become (clip a, clip b, offset) candidates, each verified exactly over the
whole overlap. For ``metric="max"`` and ``cand_factor >= 1`` this cannot miss a match (|block-mean diff| <= max
|dq|); for the mean rule the radius bounds the window RMS error (at the true alignment of the duplicated ground clips
of the RP1 test set, 65-100% of windows are within ``1.6 * tol``). Windows whose joints barely move (block-mean range
< ``min_window_motion``) are not used as candidates.
"""

from __future__ import annotations

import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

from .motion_io import load_motion
from .features import FeatureSet
from .knn import radius_pairs


def _joints(path: str, timestep: float) -> np.ndarray:
    return load_motion(path, timestep=timestep).qpos[:, 7:]


def _windows(q: np.ndarray, win: int, stride: int, blocks: int, min_motion: float):
    T, nj = q.shape
    if T < win:
        return np.zeros((0, blocks * nj), np.float32), np.zeros(0, np.int64)
    st = np.arange(0, T - win + 1, stride)
    cs = np.concatenate([np.zeros((1, nj)), np.cumsum(q, 0)])
    edges = np.linspace(0, win, blocks + 1).round().astype(int)
    x = np.stack([(cs[st + b1] - cs[st + b0]) / (b1 - b0) for b0, b1 in zip(edges[:-1], edges[1:])], 1)
    keep = (x.max(1) - x.min(1)).max(1) >= min_motion
    return x[keep].reshape(int(keep.sum()), blocks * nj).astype(np.float32), st[keep]


def _windows_job(args):
    path, timestep, win, stride, blocks, min_motion = args
    return _windows(_joints(path, timestep), win, stride, blocks, min_motion)


def _runs(ok: np.ndarray) -> list[tuple[int, int]]:
    if not ok.any():
        return []
    d = np.diff(np.concatenate([[0], ok.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _verify(qa: np.ndarray, qb: np.ndarray, delta: int, tol: float, min_len: int, metric: str, partial_len: int,
            partial_motion: float) -> list[dict]:
    """Matches between qa[t] and qb[t + delta]: a whole match if one clip lies inside the other, and partial runs."""
    t0, t1 = max(0, -delta), min(len(qa), len(qb) - delta)
    if t1 - t0 < min_len:
        return []
    dq = np.abs(qa[t0:t1] - qb[t0 + delta:t1 + delta])
    e = dq.mean(1) if metric == "mean" else dq.max(1)
    spans = []
    whole = t0 == 0 and t1 == len(qa) or t0 + delta == 0 and t1 + delta == len(qb)
    if whole and (e.mean() if metric == "mean" else e.max()) <= tol:
        spans.append((0, len(e), True))
    if t1 - t0 >= partial_len:
        if metric == "mean":
            c = np.concatenate([[0.0], np.cumsum(e)])
            good = np.flatnonzero((c[min_len:] - c[:-min_len]) / min_len <= tol)
            cover = np.zeros(len(e) + 1, np.int64)
            np.add.at(cover, good, 1)
            np.add.at(cover, good + min_len, -1)
            ok = np.cumsum(cover[:-1]) > 0
        else:
            ok = e <= tol
        spans += [(s, en, False) for s, en in _runs(ok) if en - s >= partial_len]
    out = []
    for s, en, is_whole in spans:
        seg = qa[t0 + s:t0 + en]
        motion = float((seg.max(0) - seg.min(0)).max())
        if not is_whole and motion < partial_motion:
            continue
        out.append({"a_start": int(t0 + s), "b_start": int(t0 + s + delta), "length": int(en - s),
                    "mean_abs_diff": float(dq[s:en].mean()), "max_abs_diff": float(dq[s:en].max()),
                    "frame_err_max": float(e[s:en].max()), "motion_range": motion})
    return out


def find_duplicates(A: FeatureSet, B: FeatureSet, tol: float = 0.05, min_len_s: float = 1.0, metric: str = "mean",
                    partial_min_len_s: float = 3.0, partial_min_motion: float = 1.0, blocks: int = 6,
                    cand_factor: float = 1.6, min_window_motion: float = 0.1, workers: int = 16,
                    device: str | None = None, verbose: bool = True) -> dict:
    """Duplicated segments between the clips of ``A`` and ``B`` (feature sets only provide paths / timesteps)."""
    if metric not in ("mean", "max"):
        raise ValueError("metric must be 'mean' or 'max'")
    t0 = time.time()
    dts = set(np.round(A.clip_timestep, 6)) | set(np.round(B.clip_timestep, 6))
    if len(dts) != 1:
        raise ValueError(f"dedup needs one common timestep, got {sorted(dts)}")
    dt = float(dts.pop())
    min_len = max(2, int(round(min_len_s / dt)))
    win = max(blocks, (min_len + 1) // 2)
    stride = max(1, min_len - win + 1)
    swap = int(A.clip_frames.sum()) < int(B.clip_frames.sum())
    big, small = (B, A) if swap else (A, B)

    jobs = [(p, dt, win, stride, blocks, min_window_motion) for p in big.clip_path]
    with Pool(max(1, min(workers, len(jobs)))) as pool:
        res = pool.map(_windows_job, jobs, chunksize=16)
        pool.close()
        pool.join()
    Xg = np.concatenate([r[0] for r in res])
    cg = np.concatenate([np.full(len(r[0]), i) for i, r in enumerate(res)])
    sg = np.concatenate([r[1] for r in res])
    qs = [_joints(p, dt) for p in small.clip_path]
    rs = [_windows(q, win, 1, blocks, min_window_motion) for q in qs]
    Xs = np.concatenate([r[0] for r in rs])
    cs = np.concatenate([np.full(len(r[0]), i) for i, r in enumerate(rs)])
    ss = np.concatenate([r[1] for r in rs])
    radius = max(cand_factor, 1.0 if metric == "max" else 0.0) * tol * np.sqrt(Xg.shape[1])
    mu = Xs.mean(0) if len(Xs) else 0.0
    i, j = radius_pairs(Xg - mu, Xs - mu, radius, device=device)
    cand = np.unique(np.stack([cg[i], cs[j], ss[j] - sg[i]], 1), axis=0) if len(i) else np.zeros((0, 3), np.int64)
    pair_bounds = np.flatnonzero(np.any(np.diff(cand[:, :2], axis=0) != 0, axis=1)) + 1
    if verbose:
        print(f"[dedup] windows {len(Xg)} x {len(Xs)} (win {win}, stride {stride}, {blocks} blocks x {qs[0].shape[1]} "
              f"joints, radius {radius:.2f}), pairs {len(i)}, candidates {len(cand)}, clip pairs "
              f"{len(pair_bounds) + (len(cand) > 0)} ({time.time() - t0:.0f}s)", flush=True)

    partial_len = max(min_len, int(round(partial_min_len_s / dt)))
    matches, qg, g_loaded = [], None, -1
    for block in np.split(cand, pair_bounds) if len(cand) else []:
        ga, sb = int(block[0, 0]), int(block[0, 1])
        if ga != g_loaded:
            qg, g_loaded = _joints(big.clip_path[ga], dt), ga
        found = []
        for d in block[:, 2]:
            found += _verify(qg, qs[sb], int(d), tol, min_len, metric, partial_len, partial_min_motion)
        kept = []
        for r in sorted(found, key=lambda r: (-r["length"], r["mean_abs_diff"])):
            if all(min(r["a_start"] + r["length"], k["a_start"] + k["length"]) - max(r["a_start"], k["a_start"])
                   < 0.5 * r["length"] for k in kept):
                kept.append(r)
        for r in kept:
            g_len, s_len = len(qg), len(qs[sb])
            g_s, s_s = r["a_start"], r["b_start"]
            m = {k: r[k] for k in ("length", "mean_abs_diff", "max_abs_diff", "frame_err_max", "motion_range")}
            if swap:
                ia, ib, sa, sb_, la, lb = sb, ga, s_s, g_s, s_len, g_len
            else:
                ia, ib, sa, sb_, la, lb = ga, sb, g_s, s_s, g_len, s_len
            matches.append({"a_name": str(A.clip_name[ia]), "a_group": str(A.clip_group[ia]), "a_path": str(A.clip_path[ia]),
                            "a_start": int(sa), "b_name": str(B.clip_name[ib]), "b_group": str(B.clip_group[ib]),
                            "b_path": str(B.clip_path[ib]), "b_start": int(sb_), "length_s": m["length"] * dt, **m,
                            "whole_a": m["length"] == la, "whole_b": m["length"] == lb,
                            "frac_a": m["length"] / la, "frac_b": m["length"] / lb})
    matches.sort(key=lambda m: (m["b_group"], m["b_name"], -m["length"]))
    params = {"tol": tol, "metric": metric, "min_len_s": min_len_s, "partial_min_len_s": partial_min_len_s,
              "partial_min_motion": partial_min_motion, "win": win, "stride": stride, "blocks": blocks,
              "cand_factor": cand_factor, "min_window_motion": min_window_motion, "timestep": dt, "a": A.name, "b": B.name}
    result = {"params": params, "n_candidate_window_pairs": int(len(i)), "n_candidates": int(len(cand)),
              "matches": matches, "b_clips": summarize(matches, "b"), "a_clips": summarize(matches, "a"),
              "seconds": round(time.time() - t0, 1)}
    if verbose:
        print(f"[dedup] {len(matches)} matches, {len(result['b_clips'])} {B.name} clips, "
              f"{len(result['a_clips'])} {A.name} clips ({time.time() - t0:.0f}s)", flush=True)
    return result


def summarize(matches: list[dict], side: str) -> dict:
    """Per clip of one side: matched clips of the other side, matched share of the clip, whole-segment matches."""
    other = "a" if side == "b" else "b"
    out, spans = {}, defaultdict(list)
    for m in matches:
        name = m[f"{side}_name"]
        c = out.setdefault(name, {"group": m[f"{side}_group"], "n_matches": 0, "frames": 0, "matched_frac": 0.0,
                                  f"whole_{other}_matches": 0, f"whole_{other}_clips": [], f"{other}_clips": []})
        c["n_matches"] += 1
        c["frames"] = int(round(m["length"] / m[f"frac_{side}"]))
        c[f"whole_{other}_matches"] += int(m[f"whole_{other}"])
        for key, add in ((f"{other}_clips", True), (f"whole_{other}_clips", m[f"whole_{other}"])):
            if add and m[f"{other}_name"] not in c[key]:
                c[key].append(m[f"{other}_name"])
        spans[name].append((m[f"{side}_start"], m[f"{side}_start"] + m["length"]))
    for name, iv in spans.items():
        covered = np.zeros(out[name]["frames"], bool)
        for s, e in iv:
            covered[s:e] = True
        out[name]["matched_frac"] = float(covered.mean())
    return out
