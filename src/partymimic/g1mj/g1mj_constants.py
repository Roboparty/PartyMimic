"""Unitree G1 as mjlab models it (humanoid_s0): mjlab's MJCF, actuators and FULL_COLLISION (capsule feet, self-collision
on). Its collision model is the one of the sim2real leaderboard evaluator; joints, tracked bodies, default pose, action
scale and motions are those of ``g1``."""

from dataclasses import replace

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML, get_g1_robot_cfg

from partymimic.g1.g1_constants import SPEC as G1_SPEC

SPEC = replace(G1_SPEC, name="g1mj", mjcf_path=str(G1_XML), robot_cfg=get_g1_robot_cfg)
