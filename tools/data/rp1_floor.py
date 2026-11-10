"""Floor of the RP1 KungfuAthlete clips after R2R: per clip, the 2nd percentile over frames of the lowest point of the RP1
collision geometry is moved to kungfu.FLOOR (-0.005 m, as the G1 clips) by a constant root-height shift. R2R keeps the
feet of many jump clips 2-5 cm above the ground (the BONES / LAFAN / kneel RP1 clips sit at about -0.007 m). A low
quantile, not the G1 QA's 10th percentile: the lowest point of the RP1 retargets spreads more within a clip, and the
10th percentile sinks the lowest frames of some clips up to 12 cm into the ground.
    rp1_floor.py <out.json> <dir> [...]   dirs of training npz (body_pos_w root) or any4hdmi sets (motions/*.npz qpos)
Files are rewritten in place (atomic); shifts under 1 mm are left alone, so a rerun changes nothing.
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco
import numpy as np

XML = os.environ.get("RP1_XML", str(Path(__file__).resolve().parents[2] / "data/rp1_28dof_mjlab/rp1_28dof.xml"))
FLOOR = -0.005
_M = _D = _G = _MV = None


def _init():
    global _M, _D, _G, _MV
    _M = mujoco.MjModel.from_xml_path(XML)
    _D = mujoco.MjData(_M)
    _G = [g for g in range(_M.ngeom) if _M.geom_contype[g] or _M.geom_conaffinity[g]]
    _MV = {}
    for g in _G:
        if _M.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            i = _M.geom_dataid[g]
            _MV[g] = _M.mesh_vert[_M.mesh_vertadr[i] : _M.mesh_vertadr[i] + _M.mesh_vertnum[i]]


def _lowest() -> float:
    zs = []
    for g in _G:
        p, R, s, t = _D.geom_xpos[g], _D.geom_xmat[g].reshape(3, 3), _M.geom_size[g], _M.geom_type[g]
        if t == mujoco.mjtGeom.mjGEOM_SPHERE:
            zs.append(p[2] - s[0])
        elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
            zs.append(p[2] - s[0] - abs(R[2, 2]) * s[1])
        elif t == mujoco.mjtGeom.mjGEOM_BOX:
            zs.append(p[2] - np.abs(R[2]) @ s[:3])
        elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
            zs.append(p[2] - abs(R[2, 2]) * s[1] - np.hypot(R[2, 0], R[2, 1]) * s[0])
        elif t == mujoco.mjtGeom.mjGEOM_MESH:
            zs.append((_MV[g] @ R.T)[:, 2].min() + p[2])
    return min(zs)


def fix(path: str) -> tuple[str, float]:
    if _M is None:
        _init()
    z = dict(np.load(path))
    qpos = "qpos" in z
    root = z["qpos"][:, :7] if qpos else np.concatenate((z["body_pos_w"][:, 0], z["body_quat_w"][:, 0]), -1)
    jp = z["qpos"][:, 7:] if qpos else z["joint_pos"]
    lows = []
    for t in range(len(jp)):
        _D.qpos[:7], _D.qpos[7:] = root[t], jp[t]
        mujoco.mj_kinematics(_M, _D)
        lows.append(_lowest())
    dz = FLOOR - float(np.quantile(lows, 0.02))
    if abs(dz) < 1e-3:
        return path, 0.0
    if qpos:
        z["qpos"][:, 2] += dz
    else:
        z["body_pos_w"][..., 2] += dz
    tmp = path + ".tmp.npz"
    np.savez(tmp, **z)
    os.replace(tmp, path)
    return path, dz


def main() -> None:
    files = []
    for d in map(Path, sys.argv[2:]):
        files += sorted(map(str, (d / "motions" if (d / "motions").is_dir() else d).glob("kf_*.npz")))
    with ProcessPoolExecutor(int(os.environ.get("WORKERS", 8)), initializer=_init) as ex:
        res = dict(ex.map(fix, files, chunksize=8))
    old = json.load(open(sys.argv[1])) if os.path.exists(sys.argv[1]) else {}
    old.update({k: v for k, v in res.items() if v} | {k: old.get(k, 0.0) for k, v in res.items() if not v})
    json.dump(old, open(sys.argv[1], "w"), indent=1)
    dz = np.array([v for v in res.values()])
    print(f"{len(files)} clips: shifted {(dz != 0).sum()}, dz median {np.median(dz) * 100:+.1f} cm, "
          f"min {dz.min() * 100:+.1f}, max {dz.max() * 100:+.1f}")


if __name__ == "__main__":
    main()
