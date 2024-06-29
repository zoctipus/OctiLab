from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import FrameTransformer
if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.utils.math import combine_frame_transforms


def _get_object_ee_distance(object, fixed_chop_tip_frame, free_chop_tip_frame):
    # Target object position: (num_envs, 3)
    object_pos_w = object.data.root_pos_w
    # End-effector aiming position: (num_envs, 3)
    mid_pos_w = _get_tip_mid_pos_w(fixed_chop_tip_frame, free_chop_tip_frame)
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(object_pos_w - mid_pos_w, dim=1)
    return object_ee_distance


def _get_tip_mid_pos_w(fixed_chop_tip_frame, free_chop_tip_frame):
    ee_tip_fixed_w = fixed_chop_tip_frame.data.target_pos_w[..., 0, :]
    ee_tip_free_w = free_chop_tip_frame.data.target_pos_w[..., 0, :]
    mid_pos_w = (ee_tip_fixed_w + ee_tip_free_w) / 2.0
    return mid_pos_w


def reward_object_ee_distance(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    fixed_chop_frame_cfg: SceneEntityCfg = SceneEntityCfg("frame_fixed_chop_tip"),
    free_chop_frame_cfg: SceneEntityCfg = SceneEntityCfg("frame_free_chop_tip"),
) -> torch.Tensor:
    object: RigidObject = env.scene[object_cfg.name]
    fixed_chop_frame: FrameTransformer = env.scene[fixed_chop_frame_cfg.name]
    free_chop_frame: FrameTransformer = env.scene[free_chop_frame_cfg.name]
    object_ee_distance = _get_object_ee_distance(object, fixed_chop_frame, free_chop_frame)
    return 1 - torch.tanh(object_ee_distance / 0.1)


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


'''
FRAME FUNCTIONS
'''


def _get_object_frame_ee_distance(object_frame, fixed_chop_tip_frame, free_chop_tip_frame):
    # Target object_frame position: (num_envs, 3)
    object_frame_pos_w = object_frame.data.target_pos_w[..., 0, :]
    # End-effector aiming position: (num_envs, 3)
    tip_mid_pos_w = _get_tip_mid_pos_w(fixed_chop_tip_frame, free_chop_tip_frame)
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(object_frame_pos_w - tip_mid_pos_w, dim=1)
    return object_ee_distance


def object_frame_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_frame_cfg: SceneEntityCfg = SceneEntityCfg("object_frame"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object_frame: FrameTransformer = env.scene[object_frame_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object_frame.data.target_pos_w[..., 0, :3], dim=1)
    # rewarded if the object is lifted above the threshold
    return (object_frame.data.target_pos_w[..., 0, 2] > minimal_height) * (1 - torch.tanh(distance / std))


def reward_object_frame_ee_distance(
    env: ManagerBasedRLEnv,
    object_frame_cfg: SceneEntityCfg = SceneEntityCfg("object_frame"),
    fixed_chop_frame_cfg: SceneEntityCfg = SceneEntityCfg("frame_fixed_chop_tip"),
    free_chop_frame_cfg: SceneEntityCfg = SceneEntityCfg("frame_free_chop_tip"),
) -> torch.Tensor:
    object_frame: FrameTransformer = env.scene[object_frame_cfg.name]
    fixed_chop_frame: FrameTransformer = env.scene[fixed_chop_frame_cfg.name]
    free_chop_frame: FrameTransformer = env.scene[free_chop_frame_cfg.name]
    object_ee_distance = _get_object_frame_ee_distance(object_frame, fixed_chop_frame, free_chop_frame)
    return 1 - torch.tanh(object_ee_distance / 0.1)


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
