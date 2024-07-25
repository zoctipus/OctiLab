from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import FrameTransformer
from omni.isaac.lab.utils.math import subtract_frame_transforms
if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def get_frame1_frame2_distance(frame1, frame2):
    frames_distance = torch.norm(frame1.data.target_pos_w[..., 0, :] - frame2.data.target_pos_w[..., 0, :], dim=1)
    return frames_distance


def get_body1_body2_distance(body1, body2, body1_offset, body2_offset):
    bodys_distance = torch.norm((body1.data.root_pos_w + body1_offset) - (body2.data.root_pos_w + body2_offset), dim=1)
    return bodys_distance


def get_frame1_body2_distance(frame1, body2, body2_offset):
    distance = torch.norm(frame1.data.target_pos_w[..., 0, :] - (body2.data.root_pos_w + body2_offset), dim=1)
    return distance


def reward_body_height_above(
    env: ManagerBasedRLEnv, minimum_height: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Terminate when the asset's height is below the minimum height.

    Note:
        This is currently only supported for flat terrains, i.e. the minimum height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.where(asset.data.root_pos_w[:, 2] > minimum_height, 1, 0)


def reward_frame1_frame2_distance(
    env: ManagerBasedRLEnv,
    frame1_cfg: SceneEntityCfg,
    frame2_cfg: SceneEntityCfg,
) -> torch.Tensor:
    object_frame1: FrameTransformer = env.scene[frame1_cfg.name]
    object_frame2: FrameTransformer = env.scene[frame2_cfg.name]
    frames_distance = get_frame1_frame2_distance(object_frame1, object_frame2)
    return 1 - torch.tanh(frames_distance / 0.1)


def reward_body1_body2_distance(
    env: ManagerBasedRLEnv,
    body1_cfg: SceneEntityCfg,
    body2_cfg: SceneEntityCfg,
    std: float,
    body1_offset: list[float] = [0.0, 0.0, 0.0],
    body2_offset: list[float] = [0.0, 0.0, 0.0],
) -> torch.Tensor:
    body1: RigidObject = env.scene[body1_cfg.name]
    body2: RigidObject = env.scene[body2_cfg.name]
    body1_offset_tensor = torch.tensor(body1_offset, device=env.device)
    body2_offset_tensor = torch.tensor(body2_offset, device=env.device)
    bodys_distance = get_body1_body2_distance(body1, body2, body1_offset_tensor, body2_offset_tensor)
    reward = torch.where(bodys_distance < 0.85, 5.0, 0.0)
    return 1 - torch.tanh(bodys_distance / std) + reward


def reward_body1_frame2_distance(
    env: ManagerBasedRLEnv,
    body_cfg: SceneEntityCfg,
    frame_cfg: SceneEntityCfg,
    body_offset: list[float] = [0.0, 0.0, 0.0],
) -> torch.Tensor:
    body: RigidObject = env.scene[body_cfg.name]
    object_frame: RigidObject = env.scene[frame_cfg.name]
    body_offset_tensor = torch.tensor(body_offset, device=env.device)
    bodys_distance = get_frame1_body2_distance(object_frame, body, body_offset_tensor)
    return 1 - torch.tanh(bodys_distance / 0.1)


def track_interpolated_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object"), asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    # compute the error
    obj_pos_w = obj.data.root_pos_w

    obj_pos_b, _ = subtract_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, obj_pos_w
    )
    distance = torch.norm(obj_pos_b, dim=1)
    coefficient = torch.clamp_max(distance, 1).view(-1, 1)
    vel_command = coefficient * 1 * (obj_pos_b / distance.view(-1, 1))
    env.command_manager.get_term('base_velocity').vel_command_b[:, :2] = vel_command[:, :2]
    lin_vel_error = torch.sum(
        torch.square(vel_command[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)


def track_interpolated_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object"), asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    # compute the error
    asset_pos_w = asset.data.root_pos_w
    obj_pos_w = obj.data.root_pos_w
    vel_command = torch.clamp((asset_pos_w - obj_pos_w), -1, 1)
    ang_vel_error = torch.square(vel_command[:, 2] - asset.data.root_ang_vel_b[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2-kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.where(env.episode_length_buf > 20, torch.square(asset.data.root_lin_vel_b[:, 2]), 0)
