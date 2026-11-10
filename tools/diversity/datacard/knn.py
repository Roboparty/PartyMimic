"""Chunked exact nearest-neighbour search (torch on GPU when available, numpy otherwise).

Candidates are ranked by the expanded squared distance ``|q|^2 + |r|^2 - 2 q.r`` and the selected distances are
recomputed exactly. Queries and references are both chunked, so memory stays at about
``q_chunk * r_chunk * 4`` bytes regardless of the dataset size.
"""

from __future__ import annotations

import os

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional
    torch = None


def default_device(device: str | None = None) -> str:
    if device:
        return device
    env = os.environ.get("HUMANOID_S0_DEVICE")
    if env:
        return env
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "numpy"


def knn(Q: np.ndarray, R: np.ndarray, k: int, q_group: np.ndarray | None = None, r_group: np.ndarray | None = None,
        device: str | None = None, q_chunk: int = 2048, r_chunk: int = 262144) -> tuple[np.ndarray, np.ndarray]:
    """k nearest rows of ``R`` for each row of ``Q``: (distances (n, k) float32 sorted ascending, indices (n, k)).

    Pairs with ``q_group[i] == r_group[j]`` are excluded (e.g. windows of the same clip). Missing neighbours (fewer
    than k admissible references) get distance inf and index -1.
    """
    dev = default_device(device)
    Q = np.ascontiguousarray(Q, np.float32)
    R = np.ascontiguousarray(R, np.float32)
    if (q_group is None) != (r_group is None):
        raise ValueError("q_group and r_group must be given together")
    if len(Q) == 0:
        return np.zeros((0, k), np.float32), np.zeros((0, k), np.int64)
    if len(R) == 0:
        return np.full((len(Q), k), np.inf, np.float32), np.full((len(Q), k), -1, np.int64)
    if dev == "numpy" or torch is None:
        return _knn_numpy(Q, R, k, q_group, r_group, q_chunk, min(r_chunk, 65536))
    return _knn_torch(Q, R, k, q_group, r_group, dev, q_chunk, r_chunk)


def _knn_torch(Q, R, k, qg, rg, dev, q_chunk, r_chunk):
    Rt = torch.from_numpy(R).to(dev)
    rn = (Rt * Rt).sum(1)
    rgt = torch.from_numpy(np.asarray(rg)).to(dev) if rg is not None else None
    D = np.empty((len(Q), k), np.float32)
    I = np.empty((len(Q), k), np.int64)
    for s in range(0, len(Q), q_chunk):
        q = torch.from_numpy(Q[s:s + q_chunk]).to(dev)
        qn = (q * q).sum(1, keepdim=True)
        qgt = torch.from_numpy(np.asarray(qg[s:s + q_chunk])).to(dev) if qg is not None else None
        best_d = best_i = None
        for r0 in range(0, len(R), r_chunk):
            r1 = min(r0 + r_chunk, len(R))
            d2 = qn + rn[None, r0:r1] - 2 * q @ Rt[r0:r1].T
            if qgt is not None:
                d2[qgt[:, None] == rgt[None, r0:r1]] = float("inf")
            kk = min(k, r1 - r0)
            d, i = torch.topk(d2, kk, largest=False)
            i = i + r0
            if best_d is not None:
                d, j = torch.topk(torch.cat([best_d, d], 1), min(k, best_d.shape[1] + kk), largest=False)
                i = torch.gather(torch.cat([best_i, i], 1), 1, j)
            best_d, best_i = d, i
        if best_d.shape[1] < k:
            pad = k - best_d.shape[1]
            best_d = torch.cat([best_d, torch.full((len(q), pad), float("inf"), device=dev)], 1)
            best_i = torch.cat([best_i, torch.zeros((len(q), pad), dtype=best_i.dtype, device=dev)], 1)
        exact = ((q[:, None, :] - Rt[best_i]) ** 2).sum(-1).sqrt()
        invalid = torch.isinf(best_d)
        exact[invalid] = float("inf")
        exact, order = torch.sort(exact, 1)
        idx = torch.gather(best_i, 1, order)
        idx[torch.gather(invalid, 1, order)] = -1
        D[s:s + q_chunk] = exact.cpu().numpy()
        I[s:s + q_chunk] = idx.cpu().numpy()
    return D, I


def _knn_numpy(Q, R, k, qg, rg, q_chunk, r_chunk):
    rn = (R * R).sum(1)
    D = np.empty((len(Q), k), np.float32)
    I = np.empty((len(Q), k), np.int64)
    q_chunk = min(q_chunk, 1024)
    for s in range(0, len(Q), q_chunk):
        q = Q[s:s + q_chunk]
        qn = (q * q).sum(1, keepdims=True)
        best_d = np.full((len(q), 0), np.inf, np.float32)
        best_i = np.zeros((len(q), 0), np.int64)
        for r0 in range(0, len(R), r_chunk):
            r1 = min(r0 + r_chunk, len(R))
            d2 = qn + rn[None, r0:r1] - 2 * q @ R[r0:r1].T
            if qg is not None:
                d2[np.asarray(qg[s:s + q_chunk])[:, None] == np.asarray(rg[r0:r1])[None]] = np.inf
            d = np.concatenate([best_d, d2], 1)
            i = np.concatenate([best_i, np.broadcast_to(np.arange(r0, r1), d2.shape)], 1)
            kk = min(k, d.shape[1])
            part = np.argpartition(d, kk - 1, axis=1)[:, :kk] if kk < d.shape[1] else np.argsort(d, 1)[:, :kk]
            best_d = np.take_along_axis(d, part, 1)
            best_i = np.take_along_axis(i, part, 1)
        if best_d.shape[1] < k:
            pad = k - best_d.shape[1]
            best_d = np.concatenate([best_d, np.full((len(q), pad), np.inf, np.float32)], 1)
            best_i = np.concatenate([best_i, np.zeros((len(q), pad), np.int64)], 1)
        exact = np.sqrt(((q[:, None, :] - R[best_i]) ** 2).sum(-1)).astype(np.float32)
        invalid = np.isinf(best_d)
        exact[invalid] = np.inf
        order = np.argsort(exact, 1, kind="stable")
        D[s:s + q_chunk] = np.take_along_axis(exact, order, 1)
        idx = np.take_along_axis(best_i, order, 1)
        idx[np.take_along_axis(invalid, order, 1)] = -1
        I[s:s + q_chunk] = idx
    return D, I


def nearest_center(X: np.ndarray, C: np.ndarray, device: str | None = None, chunk: int = 65536
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Index of and squared distance to the nearest row of ``C`` for each row of ``X`` (k-means assignment)."""
    dev = default_device(device)
    X = np.ascontiguousarray(X, np.float32)
    C = np.ascontiguousarray(C, np.float32)
    lab = np.empty(len(X), np.int64)
    d2o = np.empty(len(X), np.float32)
    if dev == "numpy" or torch is None:
        cn = (C * C).sum(1)
        for s in range(0, len(X), 8192):
            x = X[s:s + 8192]
            d2 = (x * x).sum(1, keepdims=True) + cn[None] - 2 * x @ C.T
            lab[s:s + 8192] = d2.argmin(1)
            d2o[s:s + 8192] = np.maximum(np.take_along_axis(d2, lab[s:s + 8192, None], 1)[:, 0], 0)
        return lab, d2o
    Ct = torch.from_numpy(C).to(dev)
    cn = (Ct * Ct).sum(1)
    for s in range(0, len(X), chunk):
        x = torch.from_numpy(X[s:s + chunk]).to(dev)
        d2 = (x * x).sum(1, keepdim=True) + cn[None] - 2 * x @ Ct.T
        m, i = d2.min(1)
        lab[s:s + chunk] = i.cpu().numpy()
        d2o[s:s + chunk] = m.clamp_min(0).cpu().numpy()
    return lab, d2o


def radius_pairs(Q: np.ndarray, R: np.ndarray, radius: float, device: str | None = None, q_chunk: int = 1024,
                 r_chunk: int = 262144, rel_slack: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Superset of all (i, j) with |Q_i - R_j| <= radius.

    The float32 expanded-distance test gets a relative slack plus an absolute term covering its rounding error, so no
    true pair is missed; callers verify candidates exactly.
    """
    dev = default_device(device)
    Q = np.ascontiguousarray(Q, np.float32)
    R = np.ascontiguousarray(R, np.float32)
    if len(Q) == 0 or len(R) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    norm_max = float((Q * Q).sum(1).max() + (R * R).sum(1).max())
    thr = radius ** 2 * (1 + rel_slack) + 1e-6 * Q.shape[1] * norm_max
    out_i, out_j = [], []
    if dev == "numpy" or torch is None:
        rn = (R * R).sum(1)
        for s in range(0, len(Q), 512):
            q = Q[s:s + 512]
            qn = (q * q).sum(1, keepdims=True)
            for r0 in range(0, len(R), 65536):
                d2 = qn + rn[None, r0:r0 + 65536] - 2 * q @ R[r0:r0 + 65536].T
                i, j = np.nonzero(d2 <= thr)
                out_i.append(i + s)
                out_j.append(j + r0)
    else:
        Rt = torch.from_numpy(R).to(dev)
        rn = (Rt * Rt).sum(1)
        for s in range(0, len(Q), q_chunk):
            q = torch.from_numpy(Q[s:s + q_chunk]).to(dev)
            qn = (q * q).sum(1, keepdim=True)
            for r0 in range(0, len(R), r_chunk):
                d2 = qn + rn[None, r0:r0 + r_chunk] - 2 * q @ Rt[r0:r0 + r_chunk].T
                ij = torch.nonzero(d2 <= thr)
                out_i.append(ij[:, 0].cpu().numpy() + s)
                out_j.append(ij[:, 1].cpu().numpy() + r0)
    if not out_i:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(out_i).astype(np.int64), np.concatenate(out_j).astype(np.int64)
