from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import FrameTransformer
if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.utils.math import combine_frame_transforms


def _get_tip_mid_pos_w(fixed_chop_tip_frame: FrameTransformer, free_chop_tip_frame: FrameTransformer):
    ee_tip_fixed_w = fixed_chop_tip_frame.data.target_pos_w[..., 0, :]
    ee_tip_free_w = free_chop_tip_frame.data.target_pos_w[..., 0, :]
    mid_pos_w = (ee_tip_fixed_w + ee_tip_free_w) / 2.0
    return mid_pos_w


def ee_frame_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    fixed_chop_frame_cfg: SceneEntityCfg = SceneEntityCfg("frame_fixed_chop_tip"),
    free_chop_frame_cfg: SceneEntityCfg = SceneEntityCfg("frame_free_chop_tip"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    fixed_chop_tip_frame: FrameTransformer = env.scene[fixed_chop_frame_cfg.name]
    free_chop_tip_frame: FrameTransformer = env.scene[free_chop_frame_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    tip_mid_pos_w = _get_tip_mid_pos_w(fixed_chop_tip_frame, free_chop_tip_frame)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - tip_mid_pos_w, dim=1)
    # rewarded if the object is lifted above the threshold
    return (1 - torch.tanh(distance / std))
