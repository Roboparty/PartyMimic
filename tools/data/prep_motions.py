"""Convert G1 clips to mjlab tracking motions at 50 Hz: joint_pos / joint_vel and the world pose and velocity of every
body, taken from MuJoCo forward kinematics of mjlab's G1 so that references and simulated bodies share one definition.
Sources: any4hdmi npz (``qpos`` [T, 36] = root xyz, root quat wxyz, 29 joints in mjlab's G1 joint order, 30 Hz),
BONES-SEED G1 csv (120 Hz) or the LAFAN1 G1 retarget csv of tools/download_data.sh (30 Hz).

  python prep_motions.py --src <dir with *.npz> --out <dir> [--fps-in 30] [--fps 50]
  python prep_motions.py --format bones --src <extracted g1/csv> --out <dir> [--list <paths>] [--workers 16]
  python prep_motions.py --format lafan --src data/lafan1_g1 --out <dir>
"""

import argparse
import glob
import json
import os

from multiprocessing import Pool

import mujoco
import numpy as np
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
from scipy.spatial.transform import Rotation


def quat_slerp(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.abs(dot).clip(max=1.0)
    theta = np.arccos(dot)
    s = np.sin(theta)
    small = s < 1e-6
    w0 = np.where(small, 1 - t, np.sin((1 - t) * theta) / np.where(small, 1, s))
    w1 = np.where(small, t, np.sin(t * theta) / np.where(small, 1, s))
    q = w0 * q0 + w1 * q1
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def resample(qpos: np.ndarray, fps_in: float, fps: float) -> np.ndarray:
    n = qpos.shape[0]
    t = np.arange(0, (n - 1) / fps_in + 1e-9, 1 / fps)
    x = t * fps_in
    i0 = np.minimum(np.floor(x).astype(int), n - 2)
    a = (x - i0)[:, None]
    out = (1 - a) * qpos[i0] + a * qpos[i0 + 1]
    out[:, 3:7] = quat_slerp(qpos[i0, 3:7], qpos[i0 + 1, 3:7], a)
    return out


def quat_to_rotvec_diff(q_prev: np.ndarray, q_next: np.ndarray) -> np.ndarray:
    """World-frame rotation vector of q_next * q_prev^-1."""
    out = np.zeros((q_prev.shape[0], 3))
    for i in range(q_prev.shape[0]):
        qi = np.zeros(4)
        mujoco.mju_negQuat(qi, q_prev[i])
        dq = np.zeros(4)
        mujoco.mju_mulQuat(dq, q_next[i], qi)
        mujoco.mju_quat2Vel(out[i], dq, 1.0)
    return out


def convert(qpos: np.ndarray, model: mujoco.MjModel, fps: float) -> dict[str, np.ndarray]:
    dt = 1 / fps
    T = qpos.shape[0]
    qvel = np.zeros((T, model.nv))
    pad = np.concatenate([qpos[:1], qpos, qpos[-1:]])
    qvel[:, :3] = (pad[2:, :3] - pad[:-2, :3]) / (2 * dt)
    qvel[:, 6:] = (pad[2:, 7:] - pad[:-2, 7:]) / (2 * dt)
    ang_w = quat_to_rotvec_diff(pad[:-2, 3:7], pad[2:, 3:7]) / (2 * dt)
    # free-joint angular velocity is expressed in the body frame
    for i in range(T):
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, qpos[i, 3:7])
        qvel[i, 3:6] = R.reshape(3, 3).T @ ang_w[i]
    qvel[[0, -1]] = qvel[[1, -2]]
    data = mujoco.MjData(model)
    nb = model.nbody - 1
    body_pos = np.zeros((T, nb, 3))
    body_quat = np.zeros((T, nb, 4))
    body_lin = np.zeros((T, nb, 3))
    body_ang = np.zeros((T, nb, 3))
    vel = np.zeros(6)
    for i in range(T):
        data.qpos[:] = qpos[i]
        data.qvel[:] = qvel[i]
        mujoco.mj_forward(model, data)
        body_pos[i] = data.xpos[1:]
        body_quat[i] = data.xquat[1:]
        for b in range(1, model.nbody):
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, b, vel, 0)
            body_ang[i, b - 1] = vel[:3]
            body_lin[i, b - 1] = vel[3:]
    return {
        "fps": np.array([fps]),
        "joint_pos": qpos[:, 7:].astype(np.float32),
        "joint_vel": qvel[:, 6:].astype(np.float32),
        "body_pos_w": body_pos.astype(np.float32),
        "body_quat_w": body_quat.astype(np.float32),
        "body_lin_vel_w": body_lin.astype(np.float32),
        "body_ang_vel_w": body_ang.astype(np.float32),
    }


def load_bones_csv(path: str) -> np.ndarray:
    """BONES-SEED G1 csv: frame, root translate (cm), root rotate (deg, extrinsic xyz = Maya rotate order xyz),
    29 joint angles (deg) in mjlab's G1 joint order."""
    a = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    qpos = np.zeros((a.shape[0], 36))
    qpos[:, :3] = a[:, 1:4] / 100
    qpos[:, 3:7] = Rotation.from_euler("xyz", a[:, 4:7], degrees=True).as_quat()[:, [3, 0, 1, 2]]
    qpos[:, 7:] = np.deg2rad(a[:, 7:36])
    return qpos


def load_lafan_csv(path: str) -> np.ndarray:
    """LAFAN1 G1 retarget csv (no header): root position (m), root quaternion xyzw, 29 joint angles (rad)."""
    a = np.loadtxt(path, delimiter=",", ndmin=2)
    return np.concatenate((a[:, :3], a[:, [6, 3, 4, 5]], a[:, 7:36]), -1)


LOADERS = {"bones": load_bones_csv, "lafan": load_lafan_csv,
           "qpos": lambda f: np.load(f)["qpos"].astype(np.float64)}
FPS_IN = {"bones": 120.0, "lafan": 30.0, "qpos": 30.0}


def _job(job):
    f, out, fmt, fps_in, fps = job
    name = os.path.splitext(os.path.basename(f))[0]
    try:
        raw = LOADERS[fmt](f)
        if raw.shape[0] < 2:
            return None
        qpos = resample(raw, fps_in, fps)
        if qpos.shape[0] < 2:
            return None
        model = mujoco.MjModel.from_xml_path(str(G1_XML))
        np.savez(os.path.join(out, name + ".npz"), **convert(qpos, model, fps))
    except (ValueError, OSError) as e:
        print("skip", name, e, flush=True)
        return None
    return {"name": name, "frames": int(qpos.shape[0])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--format", choices=sorted(LOADERS), default="qpos")
    ap.add_argument("--fps-in", type=float, default=None, help="default 30 (qpos, lafan) / 120 (bones)")
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--list", default=None, help="file with the source paths to convert (relative to --src)")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    fps_in = args.fps_in or FPS_IN[args.format]
    model = mujoco.MjModel.from_xml_path(str(G1_XML))
    os.makedirs(args.out, exist_ok=True)
    if args.list:
        files = [os.path.join(args.src, l.strip()) for l in open(args.list) if l.strip()]
    else:
        pattern = {"bones": "**/*.csv", "lafan": "*.csv", "qpos": "*.npz"}[args.format]
        files = sorted(glob.glob(os.path.join(args.src, pattern), recursive=True))
    jobs = [(f, args.out, args.format, fps_in, args.fps) for f in files]
    with Pool(args.workers) as pool:
        index = [r for r in pool.imap(_job, jobs, chunksize=4) if r is not None]
    print("converted", len(index), "of", len(files))
    json.dump({"fps": args.fps, "body_names": [model.body(i).name for i in range(1, model.nbody)], "clips": index},
              open(os.path.join(args.out, "index.json"), "w"), indent=1)
    print("clips", len(index), "hours", sum(c["frames"] for c in index) / args.fps / 3600)


if __name__ == "__main__":
    main()
