"""Robot specs of the kinematic features. MJCF: G1 from unitree_mujoco (the model the published numbers used; its key
bodies sit up to 7 mm from those of the rev-1.0 G1 of mjlab), RP1 from PartyMimic's data/rp1_28dof_mjlab
(tools/download_data.sh); override with ``DATACARD_MJCF_<NAME>`` (e.g. ``DATACARD_MJCF_G1_29DOF``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import mujoco


@dataclass(frozen=True)
class RobotSpec:
    name: str
    default_mjcf: str
    num_joints: int
    root_body: str
    # Order is fixed (L/R ankle, wrist, knee, elbow) so features stay comparable across robots.
    key_bodies: tuple[str, ...]
    feet: tuple[str, str]
    hint: str = ""

    @property
    def mjcf(self) -> str:
        mjcf = os.environ.get(f"DATACARD_MJCF_{self.name.upper()}", self.default_mjcf)
        if not os.path.exists(mjcf):
            raise FileNotFoundError(f"{mjcf} (see {self.hint})" if self.hint else mjcf)
        return mjcf

    @property
    def qpos_dim(self) -> int:
        return 7 + self.num_joints


DATA = Path(__file__).resolve().parents[3] / "data"
ROBOTS: dict[str, RobotSpec] = {
    "rp1_28dof": RobotSpec(
        name="rp1_28dof",
        default_mjcf=str(DATA / "rp1_28dof_mjlab/rp1_28dof.xml"),
        hint="tools/download_data.sh",
        num_joints=28,
        root_body="base_link",
        key_bodies=(
            "left_ankle_roll_link", "right_ankle_roll_link",
            "left_wrist_pitch_link", "right_wrist_pitch_link",
            "left_knee_link", "right_knee_link",
            "left_elbow_link", "right_elbow_link",
        ),
        feet=("left_ankle_roll_link", "right_ankle_roll_link"),
    ),
    "g1_29dof": RobotSpec(
        name="g1_29dof",
        default_mjcf=str(DATA / "unitree_mujoco_g1/g1_29dof.xml"),
        hint="tools/download_data.sh",
        num_joints=29,
        root_body="pelvis",
        key_bodies=(
            "left_ankle_roll_link", "right_ankle_roll_link",
            "left_wrist_yaw_link", "right_wrist_yaw_link",
            "left_knee_link", "right_knee_link",
            "left_elbow_link", "right_elbow_link",
        ),
        feet=("left_ankle_roll_link", "right_ankle_roll_link"),
    ),
}


def get_robot(name_or_qpos_dim: str | int) -> RobotSpec:
    if isinstance(name_or_qpos_dim, int):
        matches = [r for r in ROBOTS.values() if r.qpos_dim == name_or_qpos_dim]
        if len(matches) != 1:
            raise KeyError(f"no unique robot with qpos_dim={name_or_qpos_dim}")
        return matches[0]
    return ROBOTS[name_or_qpos_dim]


@lru_cache(maxsize=None)
def load_model(name: str) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(get_robot(name).mjcf)


def body_ids(name: str, bodies: tuple[str, ...]) -> list[int]:
    model = load_model(name)
    return [model.body(b).id for b in bodies]
