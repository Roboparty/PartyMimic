"""Sim2sim videos from the trajectories of a sim2sim evaluation (EGalahad/sim2real tracking experiment, one directory
per split with tracking_metrics.csv and the per-clip trajectory npz files): for each split the clip with the median
body position error, the robot (opaque) and the reference motion (translucent ghost) drawn at their global poses in
one scene, so root drift shows as a gap between the two. The recorded world poses of every body are written straight
into the geom poses (no joint angles needed). Run with the Python environment of sim2real:
    MUJOCO_GL=egl python tools/sim2sim_video.py <eval dir> <robot name> <robot MJCF> <out dir> \
        --sim2real <sim2real checkout> [--seconds 12]
"""

import argparse
import csv
import os
import statistics as st
import subprocess
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ctm = None  # sim2real's compute_tracking_metrics (main)
SPLITS = (("locomotion", "Locomotion"), ("manipulation", "Manipulation"), ("dance", "Dance"), ("ground", "Ground"))
GHOST = (0.25, 0.65, 1.0, 0.35)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
W, H, FPS = 960, 540, 50


def median_clip(split_dir: str) -> dict:
    rows = list(csv.DictReader(open(os.path.join(split_dir, "tracking_metrics.csv"))))
    med = st.median(float(r["local_body_tracking_error"]) for r in rows)
    r = min(rows, key=lambda r: abs(float(r["local_body_tracking_error"]) - med))
    return {"path": r["path"], "err": 1000 * float(r["local_body_tracking_error"]), "progress": float(r["progress"]),
            "end": int(r["termination_idx"]) if r["terminated"] == "1" else None}


def scene_model(xml: str) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(xml)
    tex = spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                           rgb1=[0.82, 0.84, 0.86], rgb2=[0.72, 0.74, 0.77], width=512, height=512)
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.93, 0.95, 0.98], rgb2=[0.78, 0.84, 0.92], width=512, height=3072)
    mat = spec.add_material(name="grid", texrepeat=[1, 1], reflectance=0.02)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], material="grid",
                            contype=0, conaffinity=0)
    spec.worldbody.add_light(pos=[0, 0, 5], dir=[0.3, 0.2, -1], diffuse=[0.55, 0.55, 0.55], castshadow=True,
                             type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]
    spec.visual.global_.offwidth, spec.visual.global_.offheight = W, H
    spec.visual.quality.shadowsize = 8192
    return spec.compile()


def quat_mat(q: np.ndarray) -> np.ndarray:
    """(..., 4) wxyz -> (..., 3, 3)."""
    w, x, y, z = np.moveaxis(q / np.linalg.norm(q, axis=-1, keepdims=True), -1, 0)
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                     np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                     np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


def pose_geoms(m, d, body_idx, pos, rot) -> None:
    """Geom world poses of the bodies in body_idx (model ids) from their world positions (N, 3) and rotations (N, 3, 3)."""
    for k, b in enumerate(body_idx):
        d.xpos[b], d.xmat[b] = pos[k], rot[k].reshape(-1)
    g = np.flatnonzero(m.geom_bodyid > 0)
    b = m.geom_bodyid[g]
    Rg = quat_mat(m.geom_quat[g])
    d.geom_xpos[g] = d.xpos[b] + np.einsum("nij,nj->ni", d.xmat[b].reshape(-1, 3, 3), m.geom_pos[g])
    d.geom_xmat[g] = np.einsum("nij,njk->nik", d.xmat[b].reshape(-1, 3, 3), Rg).reshape(-1, 9)


def label(img: np.ndarray, lines: list, font) -> np.ndarray:
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    for i, (txt, col) in enumerate(lines):
        dr.text((16, 12 + 26 * i), txt, fill=col, font=font, stroke_width=2, stroke_fill=(255, 255, 255))
    return np.asarray(im)


def render_clip(m, robot: str, split: str, title: str, clip: dict, out: str, seconds: float) -> dict:
    tr = dict(np.load(clip["path"]))
    idx = ctm._select_policy_frames(tr)
    names = [str(n) for n in tr["body_names"].tolist()]
    bnames = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
    use = [(bnames.index(n), j) for j, n in enumerate(names) if n in bnames]
    bid, tj = [u[0] for u in use], [u[1] for u in use]
    rp, rq, mp, mq = (np.asarray(tr[k], np.float64)[idx][:, tj] for k in
                      ("robot_body_pos_w", "robot_body_quat_w", "motion_body_pos_w", "motion_body_quat_w"))
    n = len(rp) if clip["end"] is None else max(1, clip["end"])
    n = min(n, int(seconds * FPS))
    d_rob, d_ref = mujoco.MjData(m), mujoco.MjData(m)
    mujoco.mj_forward(m, d_rob), mujoco.mj_forward(m, d_ref)
    ren = mujoco.Renderer(m, H, W)
    opt = mujoco.MjvOption()
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    rroot = np.asarray(tr["robot_root_pos_w"], np.float64)[idx]
    mroot = np.asarray(tr["motion_root_pos_w"], np.float64)[idx]
    Rm0 = quat_mat(np.asarray(tr["motion_root_quat_w"], np.float64)[idx][0])
    yaw0 = np.degrees(np.arctan2(Rm0[1, 0], Rm0[0, 0]))
    cam.azimuth, cam.elevation, cam.distance = yaw0 + 120, -12, 3.4
    font = ImageFont.truetype(FONT, 20)
    look = 0.5 * (rroot[0] + mroot[0])
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                           "-preset", "slow", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    for t in range(n):
        pose_geoms(m, d_rob, bid, rp[t], quat_mat(rq[t]))
        pose_geoms(m, d_ref, bid, mp[t], quat_mat(mq[t]))
        look = 0.95 * look + 0.05 * (0.5 * (rroot[t] + mroot[t]))
        cam.lookat[:] = [look[0], look[1], 0.55]
        ren.update_scene(d_rob, camera=cam, scene_option=opt)
        scn = ren.scene
        n0 = scn.ngeom
        mujoco.mjv_addGeoms(m, d_ref, opt, mujoco.MjvPerturb(), mujoco.mjtCatBit.mjCAT_DYNAMIC, scn)
        for i in range(n0, scn.ngeom):
            scn.geoms[i].rgba[:] = GHOST
        img = ren.render()
        err = 1000 * np.linalg.norm(rp[t] - mp[t], axis=-1).mean()
        img = label(img, [(f"PartyMimic on {robot.upper()}  |  {title} (held-out)", (20, 20, 20)),
                          (f"{os.path.basename(str(tr['motion_path'])).replace('.npz', '')}", (60, 60, 60)),
                          ("robot (solid)   reference (blue ghost, global pose)", (40, 90, 170)),
                          (f"t = {t / FPS:5.2f} s   global body position gap {err:4.0f} mm", (60, 60, 60))], font)
        ff.stdin.write(img.tobytes())
    ff.stdin.close()
    ff.wait()
    ren.close()
    return {"split": split, "clip": os.path.basename(str(tr["motion_path"])), "err_mm": round(clip["err"], 1),
            "progress": clip["progress"], "fell": clip["end"] is not None, "frames": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("lb_dir")
    ap.add_argument("robot", help="name in the title and the file names, e.g. g1, rp1")
    ap.add_argument("xml", help="the robot's MJCF (body names as in the trajectories)")
    ap.add_argument("out_dir")
    ap.add_argument("--sim2real", default=os.environ.get("SIM2REAL", "sim2real"), help="sim2real checkout")
    ap.add_argument("--seconds", type=float, default=12.0)
    a = ap.parse_args()
    global ctm
    sys.path.insert(0, os.path.join(a.sim2real, "scripts", "tracking_experiment"))
    import compute_tracking_metrics as ctm

    os.makedirs(a.out_dir, exist_ok=True)
    m = scene_model(a.xml)
    for split, title in SPLITS:
        clip = median_clip(os.path.join(a.lb_dir, split))
        info = render_clip(m, a.robot, split, title, clip, os.path.join(a.out_dir, f"{a.robot}_{split}.mp4"), a.seconds)
        print(info, flush=True)


if __name__ == "__main__":
    main()
