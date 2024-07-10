from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject, Articulation
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import FrameTransformer

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv
    from octi.lab.envs.hebi_rl_task_env import HebiRLTaskEnv
from omni.isaac.lab.utils.math import combine_frame_transforms
import octi.lab_tasks.cfgs.robots.hebi.mdp as tycho_mdp
import octi.lab.envs.mdp as general_mdp


def reward_canberry_eeframe_distance_reward(
    env: HebiRLTaskEnv,
    canberry_eeframe_distance_key: str,
):
    canberry_eeframe_distance = env.data_manager.get_active_term("data", canberry_eeframe_distance_key)
    # the closer to the canberry, higher the reward, min: 0, max 1
    return (1 - torch.tanh(canberry_eeframe_distance / 0.1)).view(-1)


def try_to_close_reward(
    env: HebiRLTaskEnv, canberry_eeframe_distance_key: str, chop_tips_canberry_cos_angle_key: str, chop_pose_key: str
):
    canberry_eeframe_distance = env.data_manager.get_active_term("data", canberry_eeframe_distance_key)
    chop_pose = env.data_manager.get_active_term("data", chop_pose_key).view(-1)
    chop_tips_canberry_cos_angle = env.data_manager.get_active_term("data", chop_tips_canberry_cos_angle_key)
    canberry_ee_proximity_mask = canberry_eeframe_distance.view(-1) < 0.03
    ball_in_chops = chop_tips_canberry_cos_angle.view(-1) < -0.4
    # if ball is within chop tips and is close enough, punish open choppose, reward close choppose, min -1, max 1
    return torch.where(canberry_ee_proximity_mask & ball_in_chops, ((-chop_pose) - 0.3) * 3, 0).clamp(-1, 1)


def miss_penalty(env: HebiRLTaskEnv, canberry_eeframe_distance_key: str, chop_pose_key: str):
    canberry_eeframe_distance = env.data_manager.get_active_term("data", canberry_eeframe_distance_key)
    chop_pose = env.data_manager.get_active_term("data", chop_pose_key)
    canberry_ee_in_range_mask = canberry_eeframe_distance < 0.05
    # if canberry is close and the end effector is completely close(< -0.57), then it is missing, min -1, max 0
    return torch.where(canberry_ee_in_range_mask & (chop_pose < -0.57), -1, 0).view(-1)


def chop_action_rate_l2(env: RLTaskEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2-kernel."""
    # punish agent switch betwen open close quickly to fake grasping: min -1, max 0
    return -torch.square((env.action_manager.action[:, -1] * 35) - (env.action_manager.prev_action[:, -1] * 35)).clamp(
        0, 1
    )


def lift_reward(env: HebiRLTaskEnv, canberry_height_key: str, canberry_grasp_mask_key: str):
    canberry_grasp_mask = env.data_manager.get_active_term("data", canberry_grasp_mask_key).view(-1)
    canberry_height = env.data_manager.get_active_term("data", canberry_height_key).view(-1)
    canberry_height_reward = torch.clamp((canberry_height - 0.013) * 20, -1, 1)
    return canberry_height_reward


def canberry_cake_distance_reward(
    env: HebiRLTaskEnv, canberry_cake_distance_key: str, canberry_height_key: str, canberry_grasp_mask_key: str
):
    canberry_grasp_mask = env.data_manager.get_active_term("data", canberry_grasp_mask_key).view(-1)
    canberry_cake_distance = env.data_manager.get_active_term("data", canberry_cake_distance_key).view(-1)
    canberry_height = env.data_manager.get_active_term("data", canberry_height_key).view(-1)
    # reward if canberry is getting close to cake while canberry is lifted, min -0.5, max 1
    return torch.where(((canberry_height > 0.03)), (1 - 1.5 * torch.tanh(canberry_cake_distance / 0.1)).view(-1), 0)


def try_to_drop_reward(env: HebiRLTaskEnv, canberry_cake_distance_key: str, chop_pose_key: str):
    canberry_cake_distance = env.data_manager.get_active_term("data", canberry_cake_distance_key)
    chop_pose = env.data_manager.get_active_term("data", chop_pose_key).view(-1)
    # the closer canberry is to the cake, we give more reward to actions that tries to open chopsticks
    return ((1 - torch.tanh(canberry_cake_distance / 0.05)).view(-1) * ((chop_pose + 0.5) * 2)).clamp(-0.01, 1)


def success_reward(
    env: HebiRLTaskEnv,
    chop_pose_key: str,
    canberry_cake_distance_key: str,
):
    chop_pose = env.data_manager.get_active_term("data", chop_pose_key)
    canberry_cake_distance = env.data_manager.get_active_term("data", canberry_cake_distance_key)

    return torch.where(torch.logical_and(canberry_cake_distance < 0.015, chop_pose > -0.4), 1, 0).view(-1)
