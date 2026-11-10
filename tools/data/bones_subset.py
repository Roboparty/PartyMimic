"""Diversity-based BONES-SEED subset (manifests/bones_seed_div10k.txt): cluster 1 s motion windows, pick clips that
fill the clusters evenly. BONES-SEED: https://huggingface.co/datasets/bones-studio/seed (g1.tar.gz, metadata/).

  python bones_subset.py feat   <extracted root (contains g1/csv)> <feat.npz> --meta <seed_metadata_v004.parquet>
  python bones_subset.py select <feat.npz> <out_list.txt> [--n 10000 --k 16384 --tau 0.5]      (CUDA)
  python bones_subset.py md     <md_feat.npz> --md <sim2real>/datasets/motiondecode              held-out clips
  python bones_subset.py cover  <feat.npz> <md_feat.npz> <list.txt>...  coverage of the pool / held-out by each list

Pool = the non-mirror clips (the mirror of a clip adds no new motion; PartyMimic mirrors in the world model), minus
the clips with the lowest body > 3 cm below the floor or a joint speed > 25 rad/s, at 50 Hz.
Window feature (10 Hz, 1 s, stride 0.5 s): 29 joint angles, root height, gravity in the root frame, root velocity in
the heading frame, yaw rate, so lying, kneeling and standing with the same joint angles are apart.
select: k-means over all pool windows; cluster k gets the target t_k ~ n_k^tau (tau < 1 flattens the common
motions); utility sum_k t_k log(1 + m_k / t_k) over the selected windows m_k (submodular), greedy by gain per cost
(clip seconds + lam). Phase 1 takes clips only from contents not yet in the set (every content_name appears once),
phase 2 is unrestricted until --n clips; --hours bisects lam so that the --n clips also match a total duration.
"""

import argparse
import os
import sys

from multiprocessing import Pool

import numpy as np

FPS, FEAT_HZ, WIN, STRIDE = 50.0, 10, 10, 5
# per-frame feature weights: joints (rad), root height (m), gravity in root frame, heading-frame velocity (m/s), yaw rate
W_Q, W_H, W_G, W_V, W_YAW = 1.0, 2.0, 1.0, 0.5, 0.3


def quat_mat(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                     2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                     2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(-1, 3, 3)


def windows(qpos):
    """qpos [T, 36] at 50 Hz -> [n_windows, WIN * 37] float32"""
    R = quat_mat(qpos[:, 3:7])
    yaw = np.unwrap(np.arctan2(R[:, 1, 0], R[:, 0, 0]))
    v = np.gradient(qpos[:, :3], 1 / FPS, axis=0)
    c, s = np.cos(yaw), np.sin(yaw)
    vh = np.stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1], v[:, 2]], -1)
    g = -R[:, 2, :]  # world -z in the root frame
    f = np.concatenate([W_Q * qpos[:, 7:], W_H * qpos[:, 2:3], W_G * g, W_V * vh,
                        W_YAW * np.gradient(yaw, 1 / FPS)[:, None]], -1)[:: int(FPS // FEAT_HZ)]
    if f.shape[0] < WIN:
        return np.zeros((0, WIN * f.shape[1]), np.float32)
    idx = np.arange(0, f.shape[0] - WIN + 1, STRIDE)[:, None] + np.arange(WIN)
    return f[idx].reshape(len(idx), -1).astype(np.float32)


_MODEL = None


def _flags(qpos):
    """lowest body z (m) and max joint speed (rad/s), as the bones10k_f filter"""
    global _MODEL
    import mujoco
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
    if _MODEL is None:
        _MODEL = mujoco.MjModel.from_xml_path(str(G1_XML))
    d = mujoco.MjData(_MODEL)
    zmin = np.inf
    for q in qpos:
        d.qpos[:] = q
        mujoco.mj_kinematics(_MODEL, d)
        zmin = min(zmin, float(d.xpos[1:, 2].min()))
    qd = np.abs(np.gradient(qpos[:, 7:], 1 / FPS, axis=0)).max() if len(qpos) > 1 else 0.0
    return zmin, float(qd)


def _feat_job(path):
    from prep_motions import load_bones_csv, resample
    try:
        raw = load_bones_csv(path)
        if raw.shape[0] < 2:
            return None
        qpos = resample(raw, 120.0, FPS)
        return windows(qpos), *_flags(qpos), qpos.shape[0] / FPS
    except (ValueError, OSError) as e:
        print("skip", path, e, flush=True)
        return None


def meta(path):
    import pyarrow.parquet as pq
    cols = ["move_g1_path", "is_mirror", "content_name", "content_type_of_movement", "package", "move_duration_frames"]
    return pq.read_table(path, columns=cols).to_pydict()


def cmd_feat(a):
    m = meta(a.meta)
    extra = {l.strip() for l in open(a.extra)} if a.extra else set()
    paths = [p for p, mir in zip(m["move_g1_path"], m["is_mirror"]) if not mir or p in extra]
    info = {p: i for i, p in enumerate(m["move_g1_path"])}
    with Pool(a.workers) as pool:
        res = list(pool.imap(_feat_job, [os.path.join(a.root, p) for p in paths], chunksize=8))
    keep = [i for i, r in enumerate(res) if r is not None]
    X = [res[i][0] for i in keep]
    np.savez(a.out, path=np.array([paths[i] for i in keep]),
             content=np.array([m["content_name"][info[paths[i]]] for i in keep]),
             mtype=np.array([m["content_type_of_movement"][info[paths[i]]] for i in keep]),
             mirror=np.array([m["is_mirror"][info[paths[i]]] for i in keep]),
             zmin=np.array([res[i][1] for i in keep]), qdmax=np.array([res[i][2] for i in keep]),
             dur=np.array([res[i][3] for i in keep]), nwin=np.array([len(x) for x in X]),
             X=np.concatenate(X).astype(np.float16))
    print("clips", len(keep), "of", len(paths), "windows", sum(len(x) for x in X))


def cmd_md(a):
    import glob
    names, split, X = [], [], []
    md = a.md.rstrip("/")
    for f in sorted(glob.glob(os.path.join(md, "*/motions/**/*.npz"), recursive=True)):
        x = windows(np.load(f)["qpos"].astype(np.float64))
        names.append(os.path.basename(f)), split.append(f[len(md) + 1:].split("/")[0]), X.append(x)
    np.savez(a.out, name=np.array(names), split=np.array(split), nwin=np.array([len(x) for x in X]),
             X=np.concatenate(X).astype(np.float16))
    print("clips", len(names), "windows", sum(len(x) for x in X))


def load_pool(path):
    F = dict(np.load(path))
    ok = (F["zmin"] > -0.03) & (F["qdmax"] <= 25.0) & (F["nwin"] > 0)
    F["ok"] = ok
    F["start"] = np.concatenate([[0], np.cumsum(F["nwin"])[:-1]])
    return F


def kmeans(X, k, iters, seed=0):
    import torch
    g = torch.Generator(device=X.device).manual_seed(seed)
    C = X[torch.randperm(len(X), generator=g, device=X.device)[:k]].clone()
    for _ in range(iters):
        lab = assign(X, C)
        S = torch.zeros_like(C).index_add_(0, lab, X)
        n = torch.bincount(lab, minlength=k).float()
        C = torch.where(n[:, None] > 0, S / n.clamp(min=1)[:, None], C)
    return C, assign(X, C)


def assign(X, C, chunk=65536):
    import torch
    out, c2 = [], (C * C).sum(1)
    for i in range(0, len(X), chunk):
        x = X[i:i + chunk]
        out.append(((x * x).sum(1, keepdim=True) - 2 * x @ C.T + c2).argmin(1))
    return torch.cat(out)


def cmd_select(a):
    import torch
    F = load_pool(a.feat)
    pool = F["ok"] & ~F["mirror"]
    clips = np.flatnonzero(pool)
    print(f"pool {pool.sum()} clips ({F['dur'][pool].sum() / 3600:.1f} h), dropped by the filter "
          f"{(~F['ok'] & ~F['mirror']).sum()}")
    win_clip = np.repeat(np.arange(len(F["nwin"])), F["nwin"])
    wmask = pool[win_clip]
    dev = "cuda"
    X = torch.tensor(F["X"][wmask], dtype=torch.float32, device=dev)
    wc = torch.tensor(np.searchsorted(clips, win_clip[wmask]), device=dev)  # window -> pool clip index
    C, lab = kmeans(X, a.k, a.iters)
    n = torch.bincount(lab, minlength=a.k).float()
    nw = torch.bincount(wc, minlength=len(clips)).float()
    budget_w = float(nw.mean()) * a.n
    t = n.clamp(min=1) ** a.tau
    t = t / t.sum() * budget_w
    dur = torch.tensor(F["dur"][clips], device=dev, dtype=torch.float32)
    content = F["content"][clips]
    cid = torch.tensor(np.unique(content, return_inverse=True)[1], device=dev)

    def greedy(lam):
        """clip cost = seconds + lam: lam = 0 is gain per second (short clips), large lam is gain per clip"""
        cost = dur + lam
        m = torch.zeros(a.k, device=dev)
        chosen = torch.zeros(len(clips), dtype=torch.bool, device=dev)
        covered = torch.zeros(int(cid.max()) + 1, dtype=torch.bool, device=dev)
        order = []
        while len(order) < a.n:
            gain_w = t[lab] * torch.log1p(1 / (t[lab] + m[lab]))  # marginal of one more window, t log(1 + m/t)
            score = torch.zeros(len(clips), device=dev).index_add_(0, wc, gain_w) / cost
            phase1 = not bool(covered.all())
            score[chosen | (covered[cid] if phase1 else torch.zeros_like(chosen))] = -1
            i = int(score.argmax())
            chosen[i] = True
            covered[cid[i]] = True
            m.index_add_(0, lab[wc == i], torch.ones(int(nw[i]), device=dev))
            order.append(i)
        return order, float(dur[chosen].sum()) / 3600

    if a.hours > 0:  # bisection on lam so that --n clips last --hours
        lo, hi = -0.95 * float(dur.min()), 4 * float(dur.mean())
        for _ in range(a.bisect):
            lam = (lo + hi) / 2
            order, h = greedy(lam)
            print(f"  lam {lam:6.2f} s -> {h:.2f} h", flush=True)
            lo, hi = (lam, hi) if h < a.hours else (lo, lam)
    else:
        order, h = greedy(float(dur.mean()))
    sel = clips[np.array(order)]
    open(a.out, "w").write("\n".join(F["path"][sel]) + "\n")
    print(f"selected {len(sel)} clips, {F['dur'][sel].sum() / 3600:.1f} h, contents {len(set(F['content'][sel]))} "
          f"of {len(set(content))}, movement types {len(set(F['mtype'][sel]))} of {len(set(F['mtype'][clips]))}")


def nn_dist(Q, R, chunk=8192):
    """RMS (per feature dim) distance of each row of Q to its nearest row of R"""
    import torch
    r2, out = (R * R).sum(1), []
    for i in range(0, len(Q), chunk):
        q = Q[i:i + chunk]
        out.append(((q * q).sum(1, keepdim=True) - 2 * q @ R.T + r2).min(1).values.clamp(min=0))
    return (torch.cat(out) / Q.shape[1]).sqrt()


def cmd_cover(a):
    import torch
    F = load_pool(a.feat)
    M = dict(np.load(a.md))
    win_clip = np.repeat(np.arange(len(F["nwin"])), F["nwin"])
    pool_w = np.flatnonzero((F["ok"] & ~F["mirror"])[win_clip])
    rng = np.random.default_rng(0)
    q_pool = torch.tensor(F["X"][rng.choice(pool_w, min(a.queries, len(pool_w)), replace=False)],
                          dtype=torch.float32, device="cuda")
    # content-balanced queries: every content_name has the same total weight (rare motions count as much as walking)
    _, inv, cnt = np.unique(F["content"][win_clip[pool_w]], return_inverse=True, return_counts=True)
    pw = 1.0 / cnt[inv]
    q_bal = torch.tensor(F["X"][rng.choice(pool_w, min(a.queries, len(pool_w)), replace=False, p=pw / pw.sum())],
                         dtype=torch.float32, device="cuda")
    q_md = torch.tensor(M["X"], dtype=torch.float32, device="cuda")
    md_split = np.repeat(M["split"], M["nwin"])
    pidx = {p: i for i, p in enumerate(F["path"])}
    print(f"{'list':28s}{'clips':>7s}{'h':>6s}{'mirror':>7s}{'contents':>9s}{'types':>6s} | pool NN rms p50/p90 "
          f"| balanced p50/p90 | held-out NN rms mean " + " ".join(f"{s[:5]:>6s}" for s in sorted(set(md_split))))
    for path in a.lists:
        rows = [pidx[l.strip()] for l in open(path) if l.strip() in pidx]
        rows = np.array([r for r in rows if F["ok"][r]])
        R = torch.tensor(F["X"][np.concatenate([np.arange(F["start"][r], F["start"][r] + F["nwin"][r]) for r in rows])],
                         dtype=torch.float32, device="cuda")
        dp, db, dm = (nn_dist(q, R).cpu().numpy() for q in (q_pool, q_bal, q_md))
        per = " ".join(f"{dm[md_split == s].mean():6.3f}" for s in sorted(set(md_split)))
        print(f"{os.path.basename(path):28s}{len(rows):7d}{F['dur'][rows].sum() / 3600:6.1f}{F['mirror'][rows].mean():7.2f}"
              f"{len(set(F['content'][rows])):9d}{len(set(F['mtype'][rows])):6d} | {np.median(dp):.3f} / "
              f"{np.quantile(dp, 0.9):.3f}        | {np.median(db):.3f} / {np.quantile(db, 0.9):.3f}    "
              f"| {dm.mean():.3f} {per}")


def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("feat"); p.add_argument("root"); p.add_argument("out")
    p.add_argument("--meta", required=True, help="BONES-SEED metadata/seed_metadata_v004.parquet")
    p.add_argument("--workers", type=int, default=24); p.add_argument("--extra", default=None)
    p = sp.add_parser("md"); p.add_argument("out"); p.add_argument("--md", required=True, help="sim2real datasets/motiondecode")
    p = sp.add_parser("select"); p.add_argument("feat"); p.add_argument("out")
    p.add_argument("--n", type=int, default=10000); p.add_argument("--k", type=int, default=16384)
    p.add_argument("--tau", type=float, default=0.5); p.add_argument("--iters", type=int, default=25)
    p.add_argument("--hours", type=float, default=0.0, help="match this total duration (0 = cost seconds + mean)")
    p.add_argument("--bisect", type=int, default=7)
    p = sp.add_parser("cover"); p.add_argument("feat"); p.add_argument("md"); p.add_argument("lists", nargs="+")
    p.add_argument("--queries", type=int, default=100000)
    a = ap.parse_args()
    {"feat": cmd_feat, "md": cmd_md, "select": cmd_select, "cover": cmd_cover}[a.cmd](a)


if __name__ == "__main__":
    main()
