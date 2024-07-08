"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`omni.isaac.lab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

"""
MDP terminations.
"""


def terminate_extremely_bad_posture(
    env: ManagerBasedRLEnv,
    probability: float = 0.5,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    robot: Articulation = env.scene[robot_cfg.name]

    elbow_position = robot.data.joint_pos[:, 2]
    shoulder_position = robot.data.joint_pos[:, 1]

    # reset for extremely bad elbow position
    elbow_punishment = torch.logical_or(elbow_position < 0.35, elbow_position > 2.9)

    # reset for extremely bad bad shoulder position
    shoulder_punishment_mask = torch.logical_or(shoulder_position < 0.1, shoulder_position > 3.0)
    bitmask = torch.rand(elbow_punishment.shape, device="cuda") < probability
    bad_posture_mask = torch.logical_or(elbow_punishment, shoulder_punishment_mask)
    return torch.where(bitmask, bad_posture_mask, False)
