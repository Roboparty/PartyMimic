"""RP1 (rp1_28dof) data: the G1 clips, retargeted G1 -> RP1 with human-humanoid-tools' R2R (tools/data/rp1_retarget.sh).

    g1csv  <npz dir> <out> --shards N    G1 npz clips (pelvis-first world body states, 50 Hz) -> MotionDecode-style
                                         G1 CSVs (R2R input), split into <out>/shard_<i>/
    md_in  <out> --shards N              the 228 MotionDecode held-out clips (G1 CSVs, 120 Hz) as R2R input shards
    npz    <rp1 csv dir> <out>           R2R RP1 CSVs -> npz clips (root position, wxyz root quaternion, joint pos in
                                         RP1 MJCF order, fps)
    testset <rp1 csv root> <out>         R2R RP1 CSVs of the 228 clips -> any4hdmi datasets <out>/<split>/ laid out like
                                         sim2real's datasets/motiondecode/<split>/ (qpos at 50 Hz, RP1 MJCF)
    hdxset <rp1 csv dir> <g1 set> <out>  RP1 copy of an any4hdmi G1 test set (the Extreme set of kungfu.py write)

--md: sim2real's datasets/motiondecode (https://github.com/EGalahad/sim2real); --md-src: the G1 CSVs of MotionDecode
(https://huggingface.co/datasets/CMRobot/MotionDecode, samples/); --xml: the RP1 MJCF (default data/rp1_28dof_mjlab,
from tools/download_data.sh). The test-set manifests store the absolute MJCF path.
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

G1_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]  # fmt: skip
REPO = Path(__file__).resolve().parents[2]
RP1_XML = str(REPO / "data/rp1_28dof_mjlab/rp1_28dof.xml")
MD_SRC = MD_DATASETS = None
SPLITS = ("locomotion", "manipulation", "ground", "dance")
OBJ_PREFIX = "x__"


def rp1_joints() -> list[str]:
    import mujoco

    m = mujoco.MjModel.from_xml_path(RP1_XML)
    return [m.joint(i).name for i in range(m.njnt) if m.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]


def g1csv(a) -> None:
    files = sorted(Path(a.src).glob("*.npz"))
    header = ["root_pos_x(m)", "root_pos_y(m)", "root_pos_z(m)", "root_rot_w", "root_rot_x", "root_rot_y", "root_rot_z"]
    header += [f"dof_{n}(rad)" for n in G1_JOINTS]
    for i, f in enumerate(files):
        d = np.load(f)
        if float(d["fps"].reshape(-1)[0]) != a.fps:
            raise SystemExit(f"{f}: fps {d['fps']} != {a.fps}")
        rows = np.concatenate((d["body_pos_w"][:, 0], d["body_quat_w"][:, 0], d["joint_pos"]), -1)
        # R2R takes a folder holding an object_*.csv for an interaction scene and skips all of its clips
        stem = f"{OBJ_PREFIX}{f.stem}" if f.stem.startswith("object_") else f.stem
        out = Path(a.out) / f"shard_{i % a.shards}" / f"{stem}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(out, rows, delimiter=",", fmt="%.6f", header=",".join(header), comments="")
    print(len(files), "clips ->", a.out)


def md_rows() -> list[dict]:
    return [r for s in SPLITS for r in csv.DictReader(open(MD_DATASETS / s / "selection_manifest.csv"))]


def md_in(a) -> None:
    for i, r in enumerate(md_rows()):
        dst = Path(a.out) / f"shard_{i % a.shards}" / r["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            os.symlink(os.path.join(MD_SRC, r["path"]), dst)
    print("->", a.out)


def load_rp1_csv(path: Path, joints: list[str]) -> tuple[np.ndarray, float]:
    """qpos rows [root xyz, root wxyz, joints in MJCF order] and the sample rate of an R2R RP1 CSV."""
    fps = None
    with open(path) as fh:
        for n_meta, line in enumerate(fh):
            if not line.startswith("#"):
                head = line.strip().split(",")
                break
            if line.startswith("# sample_rate:"):
                fps = float(line.split(":")[1])
    raw = np.loadtxt(path, delimiter=",", skiprows=n_meta + 1, ndmin=2)
    col = {n: i for i, n in enumerate(head)}
    q = raw[:, [col[k] for k in ("root_qw", "root_qx", "root_qy", "root_qz")]]
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    q *= np.where(q[:, :1] < 0, -1.0, 1.0)
    qpos = np.concatenate((raw[:, [col["root_x"], col["root_y"], col["root_z"]]], q,
                           raw[:, [col[f"dof_{n}"] for n in joints]]), -1)
    return qpos, fps


def npz(a) -> None:
    joints = rp1_joints()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n = bad = 0
    for f in sorted(Path(a.src).rglob("*.csv")):
        qpos, fps = load_rp1_csv(f, joints)
        if fps is None or len(qpos) < 2 or not np.isfinite(qpos).all():
            bad += 1
            continue
        np.savez(out / f"{f.stem.removeprefix(OBJ_PREFIX)}.npz", fps=np.array([fps]), body_pos_w=qpos[:, None, :3],
                 body_quat_w=qpos[:, None, 3:7], joint_pos=qpos[:, 7:].astype(np.float32))
        n += 1
    print(n, "clips ->", out, "| unreadable:", bad)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def testset(a) -> None:
    joints = rp1_joints()
    src, out = Path(a.src), Path(a.out)
    # R2R writes mimic clips flat; a failed clip leaves <clip>.csv.skip and possibly an empty <clip>.csv
    by_stem = {p.stem: p for p in src.rglob("*.csv") if p.stat().st_size > 0 and not p.with_suffix(".csv.skip").exists()}
    for split in SPLITS:
        g1 = MD_DATASETS / split
        man = json.loads((g1 / "manifest.json").read_text())
        root = out / split
        if root.exists():
            shutil.rmtree(root)
        rows, missing = [], []
        for r in csv.DictReader(open(g1 / "selection_manifest.csv")):
            p = by_stem.get(Path(r["path"]).stem)
            if p is None:
                missing.append(r["path"])
                continue
            qpos, fps = load_rp1_csv(p, joints)
            if abs(fps - 50.0) > 1e-6:
                raise SystemExit(f"{p}: {fps} Hz")
            rel = Path(r["motion_path"])
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            np.savez(root / rel, qpos=qpos.astype(np.float32))
            r = dict(r, frames=len(qpos), motion_sha256=_sha(root / rel))
            rows.append(r)
        if missing and not a.partial:
            raise SystemExit(f"{split}: {len(missing)} clips missing, e.g. {missing[:3]}")
        if not rows:
            continue
        man.update(dataset_name=f"motiondecode-heldout-rp1-{split}", mjcf=RP1_XML, timestep=0.02,
                   qpos_dim=7 + len(joints), qpos_names=man["qpos_names"][:7] + joints)
        (root / "manifest.json").write_text(json.dumps(man, indent=2) + "\n")
        (root / "motion-list.txt").write_text("".join(r["motion_path"] + "\n" for r in rows))
        with open(root / "selection_manifest.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        (root / "README.md").write_text(
            f"MotionDecode {split} held-out clips (same {len(rows)} clips as the G1 benchmark), retargeted G1 -> "
            "RP1 (rp1_28dof) with human-humanoid-tools R2R (profile mimic, newton, zero calibration, 50 Hz).\n")
        names = ["README.md", "manifest.json", "motion-list.txt", "selection_manifest.csv"]
        names += [r["motion_path"] for r in rows]
        (root / "checksums.json").write_text(json.dumps({n: _sha(root / n) for n in names}, indent=2) + "\n")
        print(split, len(rows))


def hdxset(a) -> None:
    """RP1 copy of an any4hdmi qpos test set (hd_holdout.py -> lb/hdx_any4hdmi) from its R2R RP1 CSVs."""
    joints = rp1_joints()
    g1, out = Path(a.g1), Path(a.out)
    by_stem = {p.stem: p for p in Path(a.src).rglob("*.csv")
               if p.stat().st_size > 0 and not p.with_suffix(".csv.skip").exists()}
    if out.exists():
        shutil.rmtree(out)
    (out / "motions").mkdir(parents=True)
    kept, missing = [], []
    for rel in (g1 / "motion-list.txt").read_text().split():
        p = by_stem.get(Path(rel).stem)
        if p is None:
            missing.append(rel)
            continue
        qpos, fps = load_rp1_csv(p, joints)
        if abs(fps - 50.0) > 1e-6:
            raise SystemExit(f"{p}: {fps} Hz")
        qpos[:, :2] -= qpos[:1, :2]
        np.savez(out / rel, qpos=qpos.astype(np.float32))
        kept.append(rel)
    man = json.loads((g1 / "manifest.json").read_text())
    man.update(mjcf=RP1_XML, qpos_dim=7 + len(joints), qpos_names=man["qpos_names"][:7] + joints,
               dataset_name=man["dataset_name"] + "-rp1", num_motions=len(kept))
    (out / "manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    (out / "motion-list.txt").write_text("".join(r + "\n" for r in kept))
    if (g1 / "split.json").exists():
        shutil.copy(g1 / "split.json", out / "split.json")
    print(len(kept), "clips ->", out, "| not retargeted:", len(missing), missing[:3])


def main() -> None:
    global RP1_XML, MD_SRC, MD_DATASETS
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("g1csv")
    p.add_argument("src")
    p.add_argument("out")
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--fps", type=float, default=50.0)
    p = sub.add_parser("md_in")
    p.add_argument("out")
    p.add_argument("--shards", type=int, default=1)
    for name in ("npz", "testset"):
        p = sub.add_parser(name)
        p.add_argument("src")
        p.add_argument("out")
    p.add_argument("--partial", action="store_true", help="skip clips not retargeted yet (smoke tests)")
    p = sub.add_parser("hdxset")
    p.add_argument("src")
    p.add_argument("g1", help="G1 any4hdmi test set")
    p.add_argument("out")
    for p in sub.choices.values():
        p.add_argument("--xml", default=RP1_XML, help="RP1 MJCF")
        p.add_argument("--md", default=None, help="sim2real datasets/motiondecode (md_in, testset)")
        p.add_argument("--md-src", default=None, help="MotionDecode G1 CSV root, samples/ (md_in)")
    a = ap.parse_args()
    RP1_XML = str(Path(a.xml).resolve())
    MD_SRC, MD_DATASETS = a.md_src, (Path(a.md) if a.md else None)
    if a.cmd in ("md_in", "testset") and MD_DATASETS is None or a.cmd == "md_in" and MD_SRC is None:
        ap.error(f"{a.cmd} needs --md" + (" and --md-src" if a.cmd == "md_in" else ""))
    {"g1csv": g1csv, "md_in": md_in, "npz": npz, "testset": testset, "hdxset": hdxset}[a.cmd](a)


if __name__ == "__main__":
    main()
