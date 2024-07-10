from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import FrameTransformer
if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def get_frame1_frame2_distance(frame1, frame2):
    frames_distance = torch.norm(frame1.data.target_pos_w[..., 0, :] - frame2.data.target_pos_w[..., 0, :], dim=1)
    return frames_distance


def get_body1_body2_distance(body1, body2, body1_offset, body2_offset):
    bodys_distance = torch.norm((body1.data.root_pos_w + body1_offset) - (body2.data.root_pos_w + body2_offset), dim=1)
    return bodys_distance


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
    body1_offset: list[float] = [0.0, 0.0, 0.0],
    body2_offset: list[float] = [0.0, 0.0, 0.0]
) -> torch.Tensor:
    body1: RigidObject = env.scene[body1_cfg.name]
    body2: RigidObject = env.scene[body2_cfg.name]
    body1_offset_tensor = torch.tensor(body1_offset, device=env.device)
    body2_offset_tensor = torch.tensor(body2_offset, device=env.device)
    bodys_distance = get_body1_body2_distance(body1, body2, body1_offset_tensor, body2_offset_tensor)
    return 1 - torch.tanh(bodys_distance / 0.1)
