"""Motion datasets in the any4hdmi layout used by mimic-lite.

A dataset directory holds ``manifest.json`` (``timestep``, ``qpos_dim``, ``num_motions``, ``source.clips``, ...)
and ``motions/NNNNN.npz`` files with ``qpos`` = [root xyz, root quat wxyz, joint angles], shape (T, 7 + nj).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Motion:
    qpos: np.ndarray
    timestep: float
    name: str
    path: str

    @property
    def num_frames(self) -> int:
        return len(self.qpos)

    @property
    def duration_s(self) -> float:
        return self.num_frames * self.timestep


@dataclass
class DatasetEntry:
    path: str
    name: str
    source: str
    group: str


MOTION_EXTS = (".npz", ".npy", ".csv", ".pkl", ".bvh", ".fbx", ".c3d", ".json", ".txt")


def strip_motion_ext(name: str) -> str:
    """Drop a known motion-file extension only (MotionDecode names such as ``1.1.2.2.Sit_to_Stand`` contain dots)."""
    for ext in MOTION_EXTS:
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return name


def source_id(clip_name: str) -> str:
    """Source clip id: segments of one long recording (``name#k``) share a source."""
    return os.path.basename(strip_motion_ext(clip_name.split("#")[0]))


def load_manifest(dataset_dir: str | Path) -> dict:
    with open(Path(dataset_dir) / "manifest.json") as f:
        return json.load(f)


def clip_names(manifest: dict) -> list[str] | None:
    src = manifest.get("source") or {}
    if isinstance(src, dict):
        for key in ("clips", "bones_seed_g1"):
            if isinstance(src.get(key), list):
                return src[key]
    return None


def list_dataset(dataset_dir: str | Path, group: str | None = None) -> list[DatasetEntry]:
    """Entries for every motion file present in ``dataset_dir/motions`` (partial copies are fine)."""
    dataset_dir = Path(dataset_dir)
    manifest = load_manifest(dataset_dir)
    names = clip_names(manifest)
    motions = dataset_dir / manifest.get("motions_subdir", "motions")
    group = group or manifest.get("dataset_name", dataset_dir.name)
    entries = []
    for f in sorted(motions.glob("*.npz")):
        idx = int(f.stem) if f.stem.isdigit() else None
        name = names[idx] if names is not None and idx is not None and idx < len(names) else f.stem
        entries.append(DatasetEntry(str(f), strip_motion_ext(name), source_id(name), group))
    return entries


def resample(qpos: np.ndarray, dt_in: float, dt_out: float) -> np.ndarray:
    """Joints and root position linear, root quaternion (wxyz) normalized lerp on a sign-continuous sequence."""
    q_in = np.asarray(qpos, dtype=np.float64).copy()
    quat = q_in[:, 3:7]
    quat[1:] *= np.cumprod(np.where(np.sum(quat[1:] * quat[:-1], axis=1) < 0, -1.0, 1.0))[:, None]
    t_in = np.arange(len(q_in)) * dt_in
    t_out = np.arange(0.0, t_in[-1] + 1e-9, dt_out)
    out = np.stack([np.interp(t_out, t_in, q_in[:, j]) for j in range(q_in.shape[1])], axis=1)
    out[:, 3:7] /= np.linalg.norm(out[:, 3:7], axis=1, keepdims=True)
    return out


def load_motion(path: str | Path, timestep: float | None = None, name: str | None = None) -> Motion:
    path = Path(path)
    if timestep is None:
        manifest_path = path.parent.parent / "manifest.json"
        timestep = json.load(open(manifest_path))["timestep"] if manifest_path.exists() else 0.02
    with np.load(path, allow_pickle=False) as z:
        qpos = np.asarray(z["qpos"], dtype=np.float64)
    return Motion(qpos=qpos, timestep=float(timestep), name=name or path.stem, path=str(path))
