"""Heading-invariant kinematic window features (robot-agnostic via ``robots``).

Per frame (``3 + 3 + 3 + 1 + 2 * nj + 3 * len(key_bodies)`` dims, 82 for RP1-24, 92 for G1-29):
root height, root linear / angular velocity in the body frame, gravity in the body frame, joint angles, joint
velocities, key bodies (L/R ankle, wrist, knee, elbow) relative to the root in the heading (yaw) frame.
Velocities are finite differences smoothed with a [1, 2, 3, 2, 1] / 9 kernel. Everything except the root height is
invariant to a global yaw rotation and an xy translation of the motion.

Per window (1 s, stride 0.5 s, frame counts derived from the clip timestep): [mean, std, last - first] of the frame
features (246 dims for RP1). Per clip: difficulty statistics of the reference motion (``DIFFICULTY_KEYS``).

``FeatureSet`` (npz) holds the windows of a dataset plus per-clip metadata; ``build_features`` extracts in parallel
and caches on a key over (feature version, robot, window settings, file list with sizes / mtimes).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from . import robots
from .motion_io import load_motion
from .datasets import ClipRef, DatasetSpec, resolve_entries, subset_names

FEATURE_VERSION = "kin_v1"
WINDOW_S = 1.0
STRIDE_S = 0.5
DIFFICULTY_KEYS = ("dur_s", "speed_p95", "speed_max", "angvel_p95", "jointvel_p99", "air_frac", "root_z_p5", "vz_p99")
AIR_MARGIN_M = 0.06
_SMOOTH = np.array([1, 2, 3, 2, 1], float) / 9
_DATA: dict[str, mujoco.MjData] = {}


def window_params(timestep: float, window_s: float = WINDOW_S, stride_s: float = STRIDE_S) -> tuple[int, int]:
    return max(1, int(round(window_s / timestep))), max(1, int(round(stride_s / timestep)))


def feature_names(robot: str) -> list[str]:
    spec = robots.get_robot(robot)
    model = robots.load_model(robot)
    joints = [model.joint(j).name for j in range(model.njnt) if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
    frame = (["root_z"] + [f"root_v{a}_b" for a in "xyz"] + [f"root_w{a}_b" for a in "xyz"] + [f"grav_{a}_b" for a in "xyz"]
             + [f"q_{j}" for j in joints] + [f"dq_{j}" for j in joints]
             + [f"{b}_{a}_h" for b in spec.key_bodies for a in "xyz"])
    return [f"{s}_{n}" for s in ("mean", "std", "delta") for n in frame]


def _smooth(x: np.ndarray) -> np.ndarray:
    p = np.pad(x, ((2, 2),) + ((0, 0),) * (x.ndim - 1), mode="edge")
    return sum(_SMOOTH[i] * p[i:i + len(x)] for i in range(5))


def _mjdata(robot: str) -> mujoco.MjData:
    if robot not in _DATA:
        _DATA[robot] = mujoco.MjData(robots.load_model(robot))
    return _DATA[robot]


def frame_features(qpos: np.ndarray, robot: str, timestep: float) -> tuple[np.ndarray, dict]:
    """Per-frame features (T, F) float32 and difficulty stats of one motion (qpos = [xyz, quat wxyz, joints])."""
    spec = robots.get_robot(robot)
    model, data = robots.load_model(robot), _mjdata(robot)
    nj = spec.num_joints
    if model.nq != 7 + nj or qpos.shape[1] != 7 + nj:
        raise ValueError(f"{robot}: qpos dim {qpos.shape[1]} / model nq {model.nq} != {7 + nj}")
    kid = robots.body_ids(robot, spec.key_bodies)
    dt = timestep
    q = qpos.astype(np.float64)
    T = len(q)
    if T < 2:
        raise ValueError("motion needs at least 2 frames")
    pos, quat, jp = q[:, :3], q[:, 3:7] / np.linalg.norm(q[:, 3:7], axis=1, keepdims=True), q[:, 7:7 + nj]
    key, minz = np.zeros((T, len(kid), 3)), np.zeros(T)
    for t in range(T):
        data.qpos[:3], data.qpos[3:7], data.qpos[7:7 + nj] = pos[t], quat[t], jp[t]
        mujoco.mj_kinematics(model, data)
        key[t] = data.xpos[kid]
        minz[t] = data.xpos[1:, 2].min()
    R = Rot.from_quat(quat[:, [1, 2, 3, 0]])
    vw = _smooth(np.gradient(pos, dt, axis=0))
    vb = R.inv().apply(vw)
    rel = (R[:-1].inv() * R[1:]).as_rotvec() / dt
    wb = _smooth(np.vstack([rel[:1], 0.5 * (rel[1:] + rel[:-1]), rel[-1:]]))
    g = R.inv().apply(np.array([0.0, 0.0, -1.0]))
    jv = _smooth(np.gradient(jp, dt, axis=0))
    fwd = R.apply(np.array([1.0, 0.0, 0.0]))
    yaw = np.arctan2(fwd[:, 1], fwd[:, 0])
    H = Rot.from_euler("z", yaw)
    krel = np.stack([H.inv().apply(key[:, i] - pos) for i in range(len(kid))], 1).reshape(T, -1)
    f = np.concatenate([pos[:, 2:3], vb, wb, g, jp, jv, krel], 1)
    speed = np.linalg.norm(vw[:, :2], axis=1)
    diff = {
        "dur_s": T * dt,
        "speed_p95": float(np.percentile(speed, 95)),
        "speed_max": float(speed.max()),
        "angvel_p95": float(np.percentile(np.linalg.norm(wb, axis=1), 95)),
        "jointvel_p99": float(np.percentile(np.abs(jv).max(1), 99)),
        "air_frac": float(np.mean(minz > np.percentile(minz, 5) + AIR_MARGIN_M)),
        "root_z_p5": float(np.percentile(pos[:, 2], 5)),
        "vz_p99": float(np.percentile(np.abs(vw[:, 2]), 99)),
    }
    return f.astype(np.float32), diff


def window_features(f: np.ndarray, window: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """[mean, std, last - first] over windows of ``window`` frames every ``stride`` frames; returns (X, start frames)."""
    T = len(f)
    if T < window:
        return np.zeros((0, 3 * f.shape[1]), np.float32), np.zeros(0, np.int64)
    st = np.arange(0, T - window + 1, stride)
    x = f[st[:, None] + np.arange(window)[None]]
    return np.concatenate([x.mean(1), x.std(1), x[:, -1] - x[:, 0]], 1).astype(np.float32), st


def clip_windows(qpos: np.ndarray, robot: str, timestep: float, window_s: float = WINDOW_S,
                 stride_s: float = STRIDE_S) -> tuple[np.ndarray, np.ndarray, dict]:
    f, diff = frame_features(qpos, robot, timestep)
    X, st = window_features(f, *window_params(timestep, window_s, stride_s))
    return X, st, diff


@dataclass
class FeatureSet:
    """Windows of a dataset (``X`` rows) and per-clip metadata (``clip_*`` arrays indexed by ``clip``)."""

    X: np.ndarray
    clip: np.ndarray
    start: np.ndarray
    clip_path: np.ndarray
    clip_name: np.ndarray
    clip_source: np.ndarray
    clip_group: np.ndarray
    clip_frames: np.ndarray
    clip_timestep: np.ndarray
    diff: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def n_clips(self) -> int:
        return len(self.clip_name)

    @property
    def n_windows(self) -> int:
        return len(self.X)

    @property
    def robot(self) -> str:
        return self.meta["robot"]

    @property
    def name(self) -> str:
        return self.meta.get("name", "dataset")

    @property
    def window_group(self) -> np.ndarray:
        return self.clip_group[self.clip]

    @property
    def clip_windows(self) -> np.ndarray:
        return np.bincount(self.clip, minlength=self.n_clips)

    @property
    def window_frames(self) -> np.ndarray:
        return np.array([window_params(t, self.meta["window_s"], self.meta["stride_s"])[0] for t in self.clip_timestep])

    @property
    def stride_frames(self) -> np.ndarray:
        return np.array([window_params(t, self.meta["window_s"], self.meta["stride_s"])[1] for t in self.clip_timestep])

    def clip_hours(self) -> np.ndarray:
        return self.clip_frames * self.clip_timestep / 3600.0

    def hours(self) -> float:
        return float(self.clip_hours().sum())

    def window_hours(self, mask: np.ndarray | None = None) -> float:
        """Hours spanned by windows at their stride (windows overlap, each one stands for ``stride_s`` seconds)."""
        n = self.n_windows if mask is None else int(np.sum(mask))
        return n * self.meta["stride_s"] / 3600.0

    def difficulty(self, key: str) -> np.ndarray:
        return self.diff[:, list(DIFFICULTY_KEYS).index(key)]

    def select_clips(self, keep: np.ndarray, name: str | None = None) -> "FeatureSet":
        keep = np.asarray(keep)
        if keep.dtype != bool:
            m = np.zeros(self.n_clips, bool)
            m[keep] = True
            keep = m
        new_idx = np.full(self.n_clips, -1, np.int64)
        new_idx[keep] = np.arange(keep.sum())
        w = keep[self.clip]
        meta = dict(self.meta, name=name or self.name)
        return FeatureSet(self.X[w], new_idx[self.clip[w]].astype(np.int32), self.start[w], self.clip_path[keep],
                          self.clip_name[keep], self.clip_source[keep], self.clip_group[keep], self.clip_frames[keep],
                          self.clip_timestep[keep], self.diff[keep], meta)

    def select_groups(self, groups, name: str | None = None) -> "FeatureSet":
        return self.select_clips(np.isin(self.clip_group, list(groups)), name)

    @staticmethod
    def concat(sets: list["FeatureSet"], name: str, description: str = "") -> "FeatureSet":
        robot = {s.robot for s in sets}
        ver = {s.meta["feature_version"] for s in sets}
        win = {(s.meta["window_s"], s.meta["stride_s"]) for s in sets}
        if len(robot) != 1 or len(ver) != 1 or len(win) != 1:
            raise ValueError(f"incompatible feature sets: robots {robot} versions {ver} windows {win}")
        off = np.cumsum([0] + [s.n_clips for s in sets])
        meta = dict(sets[0].meta, name=name, description=description, parts=[s.name for s in sets], cache_key=None)
        weights, subsets = {}, {}
        for s in sets:
            weights.update(s.meta.get("weights") or {})
            subsets.update(s.meta.get("subsets") or {})
        meta.update(weights=weights, subsets=subsets)
        cat = np.concatenate
        return FeatureSet(cat([s.X for s in sets]), cat([s.clip + o for s, o in zip(sets, off)]).astype(np.int32),
                          cat([s.start for s in sets]), cat([s.clip_path for s in sets]), cat([s.clip_name for s in sets]),
                          cat([s.clip_source for s in sets]), cat([s.clip_group for s in sets]),
                          cat([s.clip_frames for s in sets]), cat([s.clip_timestep for s in sets]),
                          cat([s.diff for s in sets]), meta)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez(tmp, X=self.X, clip=self.clip, start=self.start, clip_path=self.clip_path.astype(str),
                 clip_name=self.clip_name.astype(str), clip_source=self.clip_source.astype(str),
                 clip_group=self.clip_group.astype(str), clip_frames=self.clip_frames, clip_timestep=self.clip_timestep,
                 diff=self.diff, diff_keys=np.array(DIFFICULTY_KEYS), meta_json=np.array(json.dumps(self.meta)))
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str | Path) -> "FeatureSet":
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta_json"]))
            if tuple(z["diff_keys"]) != DIFFICULTY_KEYS:
                raise ValueError(f"{path}: difficulty keys {tuple(z['diff_keys'])} != {DIFFICULTY_KEYS}")
            return cls(z["X"], z["clip"], z["start"], z["clip_path"], z["clip_name"], z["clip_source"], z["clip_group"],
                       z["clip_frames"], z["clip_timestep"], z["diff"], meta)


def _extract_one(args):
    path, robot, timestep, window_s, stride_s = args
    q = load_motion(path, timestep=timestep).qpos
    X, st, diff = clip_windows(q, robot, timestep, window_s, stride_s)
    return X, st, [diff[k] for k in DIFFICULTY_KEYS], len(q)


def extract(entries: list[ClipRef], robot: str, workers: int = 16, window_s: float = WINDOW_S,
            stride_s: float = STRIDE_S, meta: dict | None = None, verbose: bool = True) -> FeatureSet:
    """Windows + difficulty stats for ``entries`` (multiprocessing over clips, order preserved)."""
    t0 = time.time()
    args = [(e.path, robot, e.timestep, window_s, stride_s) for e in entries]
    if workers > 1 and len(args) > 1:
        with Pool(min(workers, len(args))) as pool:
            res = pool.map(_extract_one, args, chunksize=max(1, min(8, len(args) // (4 * workers) or 1)))
            pool.close()
            pool.join()
    else:
        res = [_extract_one(a) for a in args]
    dim = len(feature_names(robot))
    X = np.concatenate([r[0] for r in res]) if res else np.zeros((0, dim), np.float32)
    clip = np.concatenate([np.full(len(r[0]), i, np.int32) for i, r in enumerate(res)]) if res else np.zeros(0, np.int32)
    start = np.concatenate([r[1] for r in res]).astype(np.int32) if res else np.zeros(0, np.int32)
    meta = dict(meta or {})
    meta.update(feature_version=FEATURE_VERSION, robot=robot, window_s=window_s, stride_s=stride_s, feature_dim=dim)
    fs = FeatureSet(X.astype(np.float32), clip, start, np.array([e.path for e in entries]),
                    np.array([e.name for e in entries]), np.array([e.source for e in entries]),
                    np.array([e.group for e in entries]), np.array([r[3] for r in res], np.int64),
                    np.array([e.timestep for e in entries], np.float64),
                    np.array([r[2] for r in res], np.float64).reshape(len(res), len(DIFFICULTY_KEYS)), meta)
    if verbose:
        print(f"[features] {meta.get('name', '')}: {fs.n_clips} clips, {fs.n_windows} windows, {fs.hours():.2f} h "
              f"({time.time() - t0:.0f}s)", flush=True)
    return fs


def infer_robot(spec: DatasetSpec, entries: list[ClipRef]) -> str:
    if spec.robot:
        return spec.robot
    dims = {e.qpos_dim for e in entries if e.qpos_dim}
    if not dims:
        with np.load(entries[0].path) as z:
            dims = {int(z["qpos"].shape[1])}
    if len(dims) != 1:
        raise ValueError(f"mixed qpos dims {dims}; set 'robot' in the spec")
    return robots.get_robot(dims.pop()).name


def cache_key(entries: list[ClipRef], robot: str, window_s: float, stride_s: float) -> str:
    h = hashlib.sha1(json.dumps([FEATURE_VERSION, robot, window_s, stride_s]).encode())
    for e in entries:
        st = os.stat(e.path)
        h.update(json.dumps([e.path, e.name, e.group, e.source, e.timestep, st.st_size, st.st_mtime_ns]).encode())
    return h.hexdigest()[:16]


def build_features(spec: DatasetSpec, out: str | Path | None = None, workers: int = 16, window_s: float = WINDOW_S,
                   stride_s: float = STRIDE_S, verbose: bool = True) -> FeatureSet:
    """Extract (or load from ``out`` if its cache key matches) the feature set of a dataset spec."""
    entries = resolve_entries(spec)
    if not entries:
        raise ValueError(f"dataset {spec.name}: no motion files found")
    robot = infer_robot(spec, entries)
    key = cache_key(entries, robot, window_s, stride_s)
    if out is not None and Path(out).exists():
        fs = FeatureSet.load(out)
        if fs.meta.get("cache_key") == key:
            if verbose:
                print(f"[features] {spec.name}: cached {out}", flush=True)
            return fs
    meta = dict(name=spec.name, description=spec.description, spec=spec.to_dict(), weights=spec.weights,
                subsets=subset_names(spec), cache_key=key)
    fs = extract(entries, robot, workers, window_s, stride_s, meta, verbose)
    if out is not None:
        fs.save(out)
    return fs
