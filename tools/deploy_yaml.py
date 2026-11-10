"""Deploy YAML of an ONNX from export_deploy_onnx.py for the EGalahad/sim2real controller (sim2sim and real robot).

Gains, action scales and default pose come from the robot constants the policy was trained with; observation groups
match the ONNX input names and use the ``supermimic.*`` observation terms of sim2real.
"""

import mujoco
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_ACTION_SCALE, G1_ARTICULATION, G1_XML, KNEES_BENT_KEYFRAME

HISTORY = 10
# fixed bodies of the evaluator's G1 model (absent from mjlab's); its body-tracking metric includes the toes
EVAL_EXTRA_BODIES = ["left_toe_link", "right_toe_link", "left_palm_link", "right_palm_link"]


def build(model_file: str, anchor: str, future: int) -> dict:
    m = mujoco.MjModel.from_xml_path(str(G1_XML))
    joints = [m.joint(i).name for i in range(m.njnt) if m.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]
    bodies = [m.body(i).name for i in range(1, m.nbody)] + EVAL_EXTRA_BODIES
    kp = {e: float(a.stiffness) for a in G1_ARTICULATION.actuators for e in a.target_names_expr}
    kd = {e: float(a.damping) for a in G1_ARTICULATION.actuators for e in a.target_names_expr}
    return {
        "model_path": model_file,
        "joint_names_simulation": joints,
        "body_names_simulation": bodies,
        "policy_joint_names": joints,
        "default_joint_pos": {k: float(v) for k, v in KNEES_BENT_KEYFRAME.joint_pos.items()},
        "action_scale": {k: float(v) for k, v in G1_ACTION_SCALE.items()},
        "joint_kp": kp,
        "joint_kd": kd,
        "motion": {
            "future_steps": list(range(-1, future)),
            "joint_names": joints,
            "body_names": ["pelvis", anchor],
            "root_body_name": "pelvis",
            "anchor_body_name": anchor,
        },
        "observation": {
            "proprio": {"proprio": {"_target_": "supermimic.proprio_history", "history_steps": HISTORY}},
            "ref_joint": {"ref_joint": {"_target_": "supermimic.ref_joint", "future_steps": list(range(future))}},
            "ref_anchor": {"ref_anchor": {"_target_": "supermimic.ref_anchor", "future_steps": list(range(-1, future))}},
            "robot_anchor_quat": {"robot_anchor_quat": {"_target_": "supermimic.robot_anchor_quat"}},
        },
    }


def build_blink(model_file: str, future: int) -> dict:
    """PartyMimic policy (export_deploy_onnx.py): same G1 constants, pelvis-rooted inputs, frames f..f+K+1."""
    cfg = build(model_file, "pelvis", future)
    steps = list(range(-1, future + 1))
    cfg["motion"].update(future_steps=steps, body_names=["pelvis"])
    cfg["observation"] = {
        "proprio": {"proprio": {"_target_": "supermimic.proprio_history", "history_steps": HISTORY}},
        "root_quat_hist": {"root_quat_hist": {"_target_": "supermimic.root_quat_history", "history_steps": HISTORY}},
        "ref_joint_pos": {"ref_joint_pos": {"_target_": "supermimic.ref_joint_pos", "future_steps": steps}},
        "ref_root": {"ref_root": {"_target_": "supermimic.ref_anchor", "future_steps": steps}},
    }
    return cfg


def build_blink_robot(model_file: str, future: int, spec) -> dict:
    """PartyMimic policy of another robot (a partymimic RobotSpec, e.g. rp1): gains, action scales, default pose and
    bodies from the spec and its MJCF; inputs as build_blink, rooted at the spec's root body."""
    import re

    m = mujoco.MjModel.from_xml_path(spec.mjcf_path)
    joints = list(spec.joint_names)
    acts = spec.robot_cfg().articulation.actuators
    gain = lambda j, attr: next(float(getattr(a, attr)) for a in acts  # noqa: E731
                                if any(re.fullmatch(p, j) for p in a.target_names_expr))
    root = spec.tracked_bodies[0]
    steps = list(range(-1, future + 1))
    return {
        "model_path": model_file,
        "joint_names_simulation": joints,
        "body_names_simulation": [m.body(i).name for i in range(1, m.nbody)],
        "policy_joint_names": joints,
        "default_joint_pos": dict(zip(joints, map(float, spec.default_joint_pos_list()))),
        "action_scale": {j: float(spec.action_scale[j]) for j in joints},
        "joint_kp": {j: gain(j, "stiffness") for j in joints},
        "joint_kd": {j: gain(j, "damping") for j in joints},
        "motion": {
            "future_steps": steps,
            "joint_names": joints,
            "body_names": [root],
            "root_body_name": root,
            "anchor_body_name": root,
        },
        "observation": {
            "proprio": {"proprio": {"_target_": "supermimic.proprio_history", "history_steps": HISTORY}},
            "root_quat_hist": {"root_quat_hist": {"_target_": "supermimic.root_quat_history",
                                                  "history_steps": HISTORY}},
            "ref_joint_pos": {"ref_joint_pos": {"_target_": "supermimic.ref_joint_pos", "future_steps": steps}},
            "ref_root": {"ref_root": {"_target_": "supermimic.ref_anchor", "future_steps": steps}},
        },
    }


def build_sparse5(model_file: str, window: int, spec, kp_names: list[str], offsets: list[int], err_dim: int) -> dict:
    """--interface sparse5 policy (export_deploy_onnx.DeploySparse5): proprio and root_quat_hist as build_blink_robot,
    the reference as ref_kp = world position + wxyz quaternion of the keypoint bodies (given order) at frames f..f+W
    (motion steps -1..W-1). ``interface`` documents the layouts the graph builds from them."""
    cfg = build_blink_robot(model_file, window, spec)
    steps = list(range(-1, window))
    cfg["motion"].update(future_steps=steps, body_names=list(kp_names))
    cfg["observation"].pop("ref_joint_pos")
    cfg["observation"].pop("ref_root")
    cfg["observation"]["ref_kp"] = {"ref_kp": {"_target_": "supermimic.ref_bodies", "future_steps": steps,
                                               "body_names": list(kp_names)}}  # fmt: skip
    cfg["interface"] = {
        "name": "sparse5",
        "keypoints": {"pelvis": kp_names[0], "head": kp_names[1], "left_wrist": kp_names[2], "right_wrist": kp_names[3],
                      "left_ankle": kp_names[4], "right_ankle": kp_names[5]},
        "ref_kp": f"[{window + 1} frames f..f+{window}, {len(kp_names)} keypoints, pos xyz (m) + quat wxyz], world frame",
        "goal_frames": list(offsets),
        "goal_per_frame": "relative to the frame-f reference pelvis heading frame: pelvis pos 3 + 6D 6, head 6D 6, "
                          "then left wrist, right wrist, left ankle, right ankle pos 3 + 6D 6 (51)",
        "error_block": f"({err_dim}) robot FK (newest joint pos + default, root quat) vs reference frame f, each relative "
                       "to its own pelvis heading frame: wrist / ankle pos error 4 x 3, so3_log ori error 4 x 3, head "
                       "so3_log ori error 3; decoder only, newest step",
        "action": "joint target = default + action_scale * action (no reference-action skip)",
    }  # fmt: skip
    return cfg
