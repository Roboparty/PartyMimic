"""RoboParty RP1 (rp1_28dof) as mjlab models it: the capsule-collision MJCF of UFO's ``rp1_28dof_mjlab`` (self-collision
on), RoboParty's current PD gains / effort limits / armatures and standing pose (robolab ``RP1_24DOF_CFG`` at 45cdc13;
its single wrist joint is our wrist roll, and the wrist yaw / pitch joints it lacks take the wrist's gains and a zero
default), joint frictionloss of rp1_description's ``mjcf/robot.xml`` (8c12d13). Tracked bodies are the RP1
counterparts of BeyondMimic's G1 set; the action scale is 0.25 * effort_limit / stiffness, as for the G1.

RP1's left / right pitch-type joint axes are mirror images (``0 1 0`` against ``0 -1 0``), so the left-right mirror
signs come from the MJCF axes instead of the G1 rule (flip roll and yaw).
"""

from __future__ import annotations

import os
import re

import mujoco
import numpy as np
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from partymimic.motion import load_npz
from partymimic.robot import RobotSpec

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MJCF_PATH = os.path.join(REPO, "data", "rp1_28dof_mjlab", "rp1_28dof.xml")
MOTION_DIR = os.path.join(REPO, "data", "rp1_motions")

# (joint regex, stiffness, damping, effort limit, armature)
ACTUATORS = (
    ("waist_(roll|yaw)_joint|.*_hip_(pitch|roll)_joint|.*_knee_joint", 120.0, 7.5, 140.0, 0.0290856),
    (".*_(hip_yaw|shoulder_pitch)_joint", 35.0, 2.5, 55.0, 0.00884),
    (".*_ankle_(pitch|roll)_joint", 25.0, 1.5, 64.0, 0.006266268),
    (".*_(shoulder_(roll|yaw)|elbow|wrist_(roll|yaw|pitch))_joint", 12.5, 0.75, 32.0, 0.00313313),
)
JOINT_FRICTIONLOSS = 0.1  # the MJCF's joint damping is already robot.xml's 0.01

TRACKED_BODIES = (
    "base_link", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "waist_yaw_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_pitch_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_pitch_link",
)  # fmt: skip

DEFAULT_JOINT_POS = {
    "left_hip_pitch_joint": -0.1, "right_hip_pitch_joint": 0.1,
    "left_knee_joint": -0.3, "right_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    "left_shoulder_pitch_joint": 0.2, "right_shoulder_pitch_joint": -0.2,
    "left_shoulder_roll_joint": 0.2, "right_shoulder_roll_joint": -0.2,
    "left_elbow_joint": -1.2, "right_elbow_joint": 1.2,
}  # fmt: skip


def actuator_of(joint: str) -> tuple[float, float, float, float]:
    for pat, *vals in ACTUATORS:
        if re.fullmatch(pat, joint):
            return tuple(vals)
    raise KeyError(joint)


def _model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(MJCF_PATH)


def _joint_names(m: mujoco.MjModel) -> tuple[str, ...]:
    return tuple(m.joint(i).name for i in range(m.njnt) if m.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE)


def mirror_signs(m: mujoco.MjModel, names: tuple[str, ...]) -> tuple[float, ...]:
    """Mirroring the world y axis (M = diag(1,-1,1)) maps a rotation q about world axis a to a rotation q about -M a;
    joint j's mirror partner k takes q_k = s q_j with s = +1 if a_k = -M a_j and -1 if a_k = M a_j (axes at qpos0)."""
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    axis = {m.joint(i).name: d.xaxis[i].copy() for i in range(m.njnt)}
    M = np.diag([1.0, -1.0, 1.0])
    signs = []
    for n in names:
        partner = n.replace("left_", "@").replace("right_", "left_").replace("@", "right_")
        a, b = M @ axis[n], axis[partner]
        if np.allclose(b, -a, atol=1e-6):
            signs.append(1.0)
        elif np.allclose(b, a, atol=1e-6):
            signs.append(-1.0)
        else:
            raise ValueError(f"{n} / {partner}: axes {axis[n]} / {b} are not mirror images")
    return tuple(signs)


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(MJCF_PATH)
    for act in list(spec.actuators):  # the position actuators come from the EntityCfg
        spec.delete(act)
    for j in spec.joints:
        if j.type == mujoco.mjtJoint.mjJNT_HINGE:
            j.frictionloss = JOINT_FRICTIONLOSS
    return spec


def get_robot_cfg() -> EntityCfg:
    acts = tuple(
        BuiltinPositionActuatorCfg(target_names_expr=(pat,), stiffness=kp, damping=kd, effort_limit=eff, armature=arm)
        for pat, kp, kd, eff, arm in ACTUATORS
    )
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.78), joint_pos=DEFAULT_JOINT_POS, joint_vel={".*": 0.0}),
        spec_fn=get_spec,
        articulation=EntityArticulationInfoCfg(actuators=acts, soft_joint_pos_limit_factor=0.9),
    )


_M = _model()
JOINT_NAMES = _joint_names(_M)

SPEC = RobotSpec(
    name="rp1",
    mjcf_path=MJCF_PATH,
    joint_names=JOINT_NAMES,
    tracked_bodies=TRACKED_BODIES,
    default_joint_pos=DEFAULT_JOINT_POS,
    action_scale={n: 0.25 * actuator_of(n)[2] / actuator_of(n)[0] for n in JOINT_NAMES},
    com_body="waist_yaw_link",
    com_range=((-0.025, 0.025), (-0.05, 0.05), (-0.05, 0.05)),
    robot_cfg=get_robot_cfg,
    motion_dir=MOTION_DIR,
    motion_glob="*.npz",
    load_motion=load_npz,
    mirror_sign=mirror_signs(_M, JOINT_NAMES),
)
