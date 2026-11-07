"""Export a PartyMimic policy checkpoint to an ONNX with raw sim2real inputs (same design as export_onnx.py).

PartyMimic's policy sees, per history step, [joint pos - default, joint vel, root angular velocity (body),
projected gravity, previous action, 6D of R_root^T R_ref_root(f)] and a goal of K future reference frames
f+1..f+K: [heading-frame root displacement, heading-frame 6D orientation, root linear / angular velocity in the
frame's own root frame, joint pos, 0.1 * joint vel] (partymimic.motion.MotionLib.goal). The graph rebuilds both
from raw inputs; reference velocities are central differences at 50 Hz, as MotionLib computes them:
    proprio          [H*93]      supermimic.proprio_history
    root_quat_hist   [H*8]       supermimic.root_quat_history: robot root quat (motion frame), reference root quat
    ref_joint_pos    [(K+2)*29]  reference joint pos at frames f..f+K+1
    ref_root         [(K+2)*7]   reference root position + wxyz quaternion at frames f..f+K+1
Run with PartyMimic's venv and src on PYTHONPATH:
    PYTHONPATH=<partymimic>/src <partymimic>/.venv/bin/python export_deploy_onnx.py <ckpt.pt> <out.onnx>
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import partymimic.networks as networks
from partymimic.kinematics import Kinematics
from partymimic.mdp import obs_dim
from partymimic.motion import CONTROL_DT, MotionLib, load_npz
from partymimic.networks import Policy, policy_kwargs
from partymimic.robot import get_robot
from partymimic.rotations import mat_to_6d, mm3, mv3, quat_to_mat, quat_to_rotvec, yaw_mat

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deploy_yaml import build_blink, build_blink_robot  # noqa: E402

SPEC = None  # the checkpoint's robot (load)
P = 93  # proprio entries per step before the 6D relative orientation: 3 J + 6


def _qmul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw), -1)


class DeployBlink(nn.Module):
    def __init__(self, pi: Policy, hist: int, K: int, J: int):
        super().__init__()
        self.pi, self.H, self.K, self.J = pi, hist, K, J
        self.register_buffer("conj", torch.tensor([1.0, -1.0, -1.0, -1.0]))

    def obs(self, proprio, root_quat_hist):
        qh = root_quat_hist.view(self.H, 8)
        rel = mat_to_6d(mm3(quat_to_mat(qh[:, :4]).transpose(-1, -2), quat_to_mat(qh[:, 4:])))
        return torch.cat((proprio.view(self.H, P), rel), -1).unsqueeze(0)

    def goal(self, ref_joint_pos, ref_root):
        K = self.K
        jp = ref_joint_pos.view(K + 2, self.J)
        rr = ref_root.view(K + 2, 7)
        p, q = rr[:, :3], rr[:, 3:]
        R = quat_to_mat(q)
        Rf = R[1 : K + 1]
        Rft = Rf.transpose(-1, -2)
        v = (p[2:] - p[:-2]) / (2 * CONTROL_DT)
        w = quat_to_rotvec(_qmul(q[2:], q[:-2] * self.conj)) / (2 * CONTROL_DT)
        Ht = yaw_mat(R[0]).transpose(-1, -2).unsqueeze(0)
        dp = mv3(Ht, p[1 : K + 1] - p[0:1])
        rot = mat_to_6d(mm3(Ht, Rf))
        qd = (jp[2:] - jp[:-2]) / (2 * CONTROL_DT)
        return torch.cat((dp, rot, mv3(Rft, v), mv3(Rft, w), jp[1 : K + 1], qd * 0.1), -1).reshape(1, -1)

    def forward(self, proprio, root_quat_hist, ref_joint_pos, ref_root):
        return self.pi(self.obs(proprio, root_quat_hist), self.goal(ref_joint_pos, ref_root)).reshape(self.J)


def load(ckpt: str) -> tuple[Policy, dict]:
    global SPEC, P
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    a = ck["args"]
    SPEC = get_robot(a.get("robot", "g1"))
    P = 3 * len(SPEC.joint_names) + 6
    if a.get("wm_contact", 0) or a.get("pi_arch", "mlp") != "mlp":
        raise SystemExit("unsupported policy architecture")
    widths = lambda key, default: tuple(int(x) for x in a.get(key, default).split(","))  # noqa: E731
    J = len(SPEC.joint_names)
    K = a["goal_frames"]
    pi = Policy(obs_dim(J), K * (15 + 2 * J), J, hist=a["hist"], enc_hidden=widths("enc_hidden", "2048,1024,512,256"),
                dec_hidden=widths("dec_hidden", "2048,1024,512,256"), **policy_kwargs(a, SPEC))
    pi.load_state_dict(ck["pi"])
    return pi.eval(), a


def window(motion: MotionLib, f: int, K: int):
    """Raw ref_joint_pos / ref_root of frame f, frames clamped to the clip like sim2real's slicing."""
    end = int(motion.clip_end[motion.clip_id[f]])
    idx = torch.clamp(torch.arange(f, f + K + 2), max=end)
    from partymimic.rotations import mat_to_quat

    root = torch.cat((motion.root_p[idx], mat_to_quat(motion.root_R[idx])), -1)
    return motion.q[idx].reshape(-1), root.reshape(-1)


def check_geometry(dep: DeployBlink, motion: MotionLib, n: int = 200) -> float:
    """Graph goal against MotionLib.goal on frames at least two away from the clip end (where MotionLib switches to
    one-sided differences)."""
    g = torch.Generator().manual_seed(0)
    ok = (motion.clip_end[motion.clip_id] - torch.arange(motion.num_frames)) > dep.K + 1
    ok &= torch.arange(motion.num_frames) > motion.clip_start[motion.clip_id]
    frames = ok.nonzero().flatten()
    frames = frames[torch.randint(len(frames), (n,), generator=g)]
    err = 0.0
    for f in frames.tolist():
        jp, rr = window(motion, f, dep.K)
        ref = motion.goal(torch.tensor([f]), dep.K)
        err = max(err, (dep.goal(jp, rr) - ref).abs().max().item())
    return err


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out")
    a = ap.parse_args()
    networks.AMP_DTYPE = None
    pi, args = load(a.ckpt)
    J, K, H = len(SPEC.joint_names), args["goal_frames"], args["hist"]
    dep = DeployBlink(pi, H, K, J).eval()

    kin = Kinematics(SPEC, "cpu")
    # a few clips of the training library are enough for the checks
    md = Path(args.get("motion_dir") or SPEC.motion_dir)
    if not md.is_dir():  # trained on another machine
        md = Path(os.environ.get("MOTION_DIR", SPEC.motion_dir))
    files, loader = sorted(md.glob(SPEC.motion_glob))[:8], SPEC.load_motion
    if not files:
        files, loader = sorted(md.glob("*.npz"))[:8], load_npz
    if not files:
        raise SystemExit(f"no motions in {md} for the export checks; set MOTION_DIR")
    motion = MotionLib([str(f) for f in files], kin, "cpu", loader)
    with torch.no_grad():
        geo = check_geometry(dep, motion)
        # relative orientation and the full policy against Policy(obs_hist, MotionLib.goal)
        g = torch.Generator().manual_seed(1)
        f = int(motion.clip_start[3]) + 100
        jp, rr = window(motion, f, K)
        prop = torch.randn(H * P, generator=g) * 0.3
        rq = torch.nn.functional.normalize(torch.randn(H, 4, generator=g), dim=-1)
        from partymimic.rotations import mat_to_quat

        rq = torch.cat((rq, mat_to_quat(motion.root_R[f]).expand(H, 4)), -1).reshape(-1)
        Rr = quat_to_mat(rq.view(H, 8)[:, :4])
        rel_ref = mat_to_6d(mm3(Rr.transpose(-1, -2), motion.root_R[f]))
        obs_ref = torch.cat((prop.view(H, P), rel_ref), -1).unsqueeze(0)
        pol_err = (dep(prop, rq, jp, rr) - pi(obs_ref, motion.goal(torch.tensor([f]), K))[0]).abs().max().item()

    inputs = (prop, rq, jp, rr)
    names = ["proprio", "root_quat_hist", "ref_joint_pos", "ref_root"]
    out = Path(a.out)
    torch.onnx.export(dep, inputs, str(out), input_names=names, output_names=["action"], opset_version=17,
                      dynamo=False)
    np.savez(out.with_suffix(".check.npz"), **{n: x.numpy() for n, x in zip(names, inputs)},
             action=dep(*inputs).detach().numpy())
    cfg = build_blink(out.name, K) if SPEC.name in ("g1", "g1mj") else build_blink_robot(out.name, K, SPEC)
    out.with_suffix(".yaml").write_text(__import__("yaml").safe_dump(cfg, sort_keys=False))
    print(json.dumps({"onnx": str(out), "robot": SPEC.name, "future": K, "history": H, "geometry_max_err": geo,
                      "policy_vs_training_max_err": pol_err, "inputs": names}))


if __name__ == "__main__":
    main()
