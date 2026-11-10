"""KungfuAthlete (https://github.com/NPCLEI/KungFuAthleteBot, video -> GVHMR -> GMR, smoothed G1 qpos at 50 Hz) as
training data and the Extreme test set.

    qa <raw org_smooth dir> <titles.csv> <out.json>
        per clip: subset (ground / jump), source video and style, duration, floor offset and ground penetration of the
        collision geometry, self-penetration, joint-limit excess and joint / root speed, from MuJoCo FK of mjlab's G1
    split <qa.json> <out.json> [--test 0.3]
        quality filter, then a held-out test share drawn per (subset, style) stratum by whole source videos, so that
        no test clip has a sibling cut from the same video in training (manifests/kungfu_split.json is its output)
    write <qa.json> <split.json> <out>
        floor-corrected 50 Hz qpos: <out>/train_q50/*.npz (prep_motions.py --fps-in 50 input) and the Extreme test set
        <out>/test_any4hdmi for the sim2real evaluator (root xy zeroed at the first frame)
"""
import argparse
import csv
import json
import re
import shutil
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np

STYLES = [  # first match wins; titles are the source video names of 'Action File Title.csv'
    ("flip", r"空翻|后手翻|侧手翻|旋子|旋风|腾空|摆莲|翻身"),
    ("taiji", r"太极"),
    ("staff", r"棍|枪"),
    ("saber_sword", r"刀|剑"),
    ("fist", r"拳|掌"),
]
LIMIT_SLACK = 0.05
G1_JOINTS = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint",
    "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint", "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint")  # fmt: skip
# any4hdmi dataset manifest of the sim2real evaluator (its G1 MJCF, qpos layout, 50 Hz)
MANIFEST = {"format_version": 2, "mjcf": "hf://elijahgalahad/g1_xmls@main/g1-mode_13_15.xml", "motions_subdir": "motions",
            "qpos_dim": 36, "qpos_names": ["root_tx", "root_ty", "root_tz", "root_qw", "root_qx", "root_qy", "root_qz",
                                           *G1_JOINTS]}
_M = None


def model() -> mujoco.MjModel:
    global _M
    if _M is None:
        from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
        _M = mujoco.MjModel.from_xml_path(str(G1_XML))
    return _M


def style(title: str) -> str:
    for name, rx in STYLES:
        if re.search(rx, title):
            return name
    return "training"


def video(title: str) -> str:
    return re.sub(r"-Scene-\d+\.mp4$", "", title)


def _lowest(m: mujoco.MjModel, d: mujoco.MjData, geoms: np.ndarray) -> float:
    z = np.inf
    for g in geoms:
        p, r = d.geom_xpos[g], m.geom_size[g]
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
            z = min(z, p[2] - r[0])
        else:  # capsule
            ax = d.geom_xmat[g].reshape(3, 3)[:, 2]
            z = min(z, p[2] - abs(ax[2]) * r[1] - r[0])
    return z


def _qa(job):
    path, sub = job
    m = model()
    d = mujoco.MjData(m)
    f = np.load(path)
    q, fps = f["qpos"].astype(np.float64), float(f["fps"])
    geoms = np.array([g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]])
    low, selfpen = [], []
    for t in range(0, len(q), 2):
        d.qpos[:] = q[t]
        mujoco.mj_forward(m, d)
        low.append(_lowest(m, d, geoms))
        pen = [-d.contact[i].dist for i in range(d.ncon)]
        selfpen.append(max(pen, default=0.0))
    low, selfpen = np.array(low), np.array(selfpen)
    lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]
    excess = np.maximum(lo - q[:, 7:], q[:, 7:] - hi).max(1)
    jv = np.abs(np.diff(q[:, 7:], axis=0)).max(1) * fps
    rv = np.linalg.norm(np.diff(q[:, :3], axis=0), axis=1) * fps
    ok = np.isfinite(q).all()
    return {
        "id": Path(path).stem, "subset": sub, "path": str(path), "frames": len(q), "fps": fps,
        "dur": len(q) / fps, "finite": bool(ok),
        # floor: the 10th percentile of the per-frame lowest point (a clip on the ground has it near 0)
        "floor": float(np.percentile(low, 10)), "low_min": float(low.min()),
        "pen_frac": float((low < -0.03).mean()), "float_frac": float((low > 0.05).mean()),
        "self_pen_max": float(selfpen.max()), "self_pen_frac": float((selfpen > 0.02).mean()),
        "limit_excess": float(excess.max()), "limit_frac": float((excess > LIMIT_SLACK).mean()),
        "jv_max": float(jv.max()), "jv_p99": float(np.percentile(jv, 99)), "root_v_max": float(rv.max()),
    }


def qa(a) -> None:
    titles = {r["Data Number"].removesuffix(".mp4"): r["Source Video Title"] for r in csv.DictReader(open(a.titles, encoding="utf-8-sig"))}
    jobs = [(str(p), "ground" if "ground" in p.parent.parent.name else "jump")
            for p in sorted(Path(a.raw).glob("*/*/*.npz"))]
    with Pool(a.workers) as pool:
        rows = pool.map(_qa, jobs, chunksize=4)
    for r in rows:
        t = titles.get(r["id"], "")
        r.update(title=t, video=video(t), style=style(t))
    json.dump(rows, open(a.out, "w"), ensure_ascii=False, indent=0)
    print(len(rows), "clips ->", a.out)


FLOOR = -0.005      # floor (qa) of the BONES / LAFAN / kneel training clips: the lowest geometry ~5 mm into the ground
MIN_DUR = 1.0       # s
MAX_LIMIT_FRAC = 0.15
SEG, SEG_MIN = 20.0, 5.0   # clips longer than SEG + SEG_MIN s are cut into SEG s pieces (a short tail joins the last)


def segments(n: int, fps: float) -> list[tuple[int, int]]:
    seg, tail = int(SEG * fps), int(SEG_MIN * fps)
    if n <= seg + tail:
        return [(0, n)]
    cuts = list(range(0, n, seg))
    if n - cuts[-1] < tail:
        cuts.pop()
    return [(s, cuts[i + 1] if i + 1 < len(cuts) else n) for i, s in enumerate(cuts)]


def split(a) -> None:
    rows = json.load(open(a.qa))
    keys, drop = {}, {"short": [], "limits": [], "duplicate": [], "nonfinite": []}
    keep = []
    for r in rows:
        if not r["finite"]:
            drop["nonfinite"].append(r["id"])
        elif r["dur"] < MIN_DUR:
            drop["short"].append(r["id"])
        elif r["limit_frac"] > MAX_LIMIT_FRAC:
            drop["limits"].append(r["id"])
        else:
            q = np.load(r["path"])["qpos"]
            k = np.round(q[:, 7:], 3).astype(np.float32).tobytes()
            if k in keys:
                drop["duplicate"].append(r["id"])
                continue
            keys[k] = r["id"]
            keep.append(r)
    for r in keep:
        r["stratum"] = f"{r['subset']}/{r['style']}"
        r["n_seg"] = len(segments(r["frames"], r["fps"]))
    by_video: dict[str, list[dict]] = {}
    for r in keep:
        by_video.setdefault(r["video"], []).append(r)
    # a video belongs to the stratum most of its segments are in; whole videos go to test until a stratum's test
    # share reaches the target
    strata: dict[str, list[str]] = {}
    for v, rs in by_video.items():
        cnt: dict[str, int] = {}
        for r in rs:
            cnt[r["stratum"]] = cnt.get(r["stratum"], 0) + r["n_seg"]
        strata.setdefault(max(cnt, key=cnt.get), []).append(v)
    rng = np.random.default_rng(a.seed)
    test_videos = set()
    for s, vids in sorted(strata.items()):
        total = sum(r["n_seg"] for v in vids for r in by_video[v])
        target, got = a.test * total, 0
        for i in rng.permutation(len(vids)):
            v = vids[i]
            n = sum(r["n_seg"] for r in by_video[v])
            if got == 0 or abs(got + n - target) < abs(got - target):
                test_videos.add(v)
                got += n
            if got >= target:
                break
        if len(vids) < 2:  # a stratum of one video cannot be split by video: keep it for training
            test_videos -= set(vids)
    out = {"test": sorted(r["id"] for r in keep if r["video"] in test_videos),
           "train": sorted(r["id"] for r in keep if r["video"] not in test_videos),
           "test_videos": sorted(test_videos), "dropped": drop, "seed": a.seed, "test_share": a.test}
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    idx = {r["id"]: r for r in keep}
    for part in ("train", "test"):
        rs = [idx[i] for i in out[part]]
        st: dict[str, list] = {}
        for r in rs:
            st.setdefault(r["stratum"], [0, 0.0])
            st[r["stratum"]][0] += r["n_seg"]
            st[r["stratum"]][1] += r["dur"]
        print(part, len(rs), "clips", sum(r["n_seg"] for r in rs), "segments", f"{sum(r['dur'] for r in rs) / 3600:.2f} h",
              {k: (v[0], round(v[1] / 60, 1)) for k, v in sorted(st.items())})
    print("dropped", {k: len(v) for k, v in drop.items()})


def clean(q: np.ndarray, floor: float) -> np.ndarray:
    m = model()
    q = q.astype(np.float64).copy()
    q[:, 2] += FLOOR - floor
    q[:, 7:] = np.clip(q[:, 7:], m.jnt_range[1:, 0], m.jnt_range[1:, 1])
    q[:, 3:7] /= np.linalg.norm(q[:, 3:7], axis=1, keepdims=True)
    return q


def write(a) -> None:
    rows = {r["id"]: r for r in json.load(open(a.qa))}
    sp = json.load(open(a.split))
    out = Path(a.out)
    for d in ("train_q50", "test_any4hdmi"):
        if (out / d).exists():
            shutil.rmtree(out / d)
    (out / "train_q50").mkdir(parents=True)
    (out / "test_any4hdmi" / "motions").mkdir(parents=True)
    names = {"train": [], "test": []}
    for part in ("train", "test"):
        for i in sp[part]:
            r = rows[i]
            q = clean(np.load(r["path"])["qpos"], r["floor"])
            segs = segments(len(q), r["fps"])
            for j, (s, e) in enumerate(segs):
                n = f"kf_{r['subset']}_{r['style']}_{i}" + (f"_p{j}" if len(segs) > 1 else "")
                qs = q[s:e].copy()
                if part == "train":
                    np.savez(out / "train_q50" / f"{n}.npz", qpos=qs, fps=np.array(50.0))
                else:
                    qs[:, :2] -= qs[:1, :2]
                    np.savez(out / "test_any4hdmi" / "motions" / f"{n}.npz", qpos=qs.astype(np.float32))
                names[part].append(n)
    man = dict(MANIFEST, dataset_name="kungfu-athlete-holdout", timestep=0.02, num_motions=len(names["test"]))
    t = out / "test_any4hdmi"
    (t / "manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    (t / "motion-list.txt").write_text("".join(f"motions/{n}.npz\n" for n in names["test"]))
    (t / "split.json").write_text(json.dumps({"test": names["test"], "train": names["train"]}, ensure_ascii=False))
    print({k: len(v) for k, v in names.items()}, "->", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("qa")
    p.add_argument("raw")
    p.add_argument("titles")
    p.add_argument("out")
    p.add_argument("--workers", type=int, default=12)
    p = sub.add_parser("split")
    p.add_argument("qa")
    p.add_argument("out")
    p.add_argument("--test", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("write")
    p.add_argument("qa")
    p.add_argument("split")
    p.add_argument("out")
    a = ap.parse_args()
    {"qa": qa, "split": split, "write": write}[a.cmd](a)


if __name__ == "__main__":
    main()
