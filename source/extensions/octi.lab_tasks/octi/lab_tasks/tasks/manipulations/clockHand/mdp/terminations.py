from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor


if TYPE_CHECKING:
    from octi.lab.envs.hebi_rl_task_env import HebiRLTaskEnv
    from omni.isaac.lab.managers.command_manager import CommandTerm
import octi.lab_tasks.cfgs.robots.hebi.mdp as tycho_mdp
from octi.lab.managers.data_manager import History
import octi.lab.envs.mdp as general_mdp

"""
MDP terminations.
"""

# def chop_missing(
#     env: HebiRLTaskEnv,
#     robot_name:str = "robot",
#     canberry_name: str = "canberry",
#     caneberry_offset: list[float] = [0.0, 0.0, 0.0],
#     fixed_chop_frame_name = "frame_fixed_chop_tip",
#     free_chop_frame_name = "frame_free_chop_tip",
# ):
#     robot: Articulation = env.scene[robot_name]
#     canberry: RigidObject = env.scene[canberry_name]
#     fixed_chop_frame = env.scene[fixed_chop_frame_name]
#     free_chop_frame = env.scene[free_chop_frame_name]
#     canberry_offset_tensor = torch.tensor(caneberry_offset, device = env.device)
#     chop_pose = robot.data.joint_pos[:, -1]
#     canberry_eeframe_distance = tycho_mdp.get_object_eeFrame_distance(canberry, canberry_offset_tensor, fixed_chop_frame, free_chop_frame)
#     missing_mask = (chop_pose < -0.60) & (canberry_eeframe_distance < 0.035)
#     return missing_mask


def chop_missing(
    env: HebiRLTaskEnv,
    canberry_eeframe_distance_key: str,
    chop_pose_key: str,
):
    canberry_eeframe_distance = env.data_manager.get_active_term("data", canberry_eeframe_distance_key)
    chop_pose = env.data_manager.get_active_term("data", chop_pose_key)
    return ((chop_pose < -0.57) & (canberry_eeframe_distance < 0.035)).view(-1)


def non_moving_abnormalty(
    env: HebiRLTaskEnv, end_effector_speed_str: str = "end_effector_speed", ee_position_b_str: str = "ee_position_b"
):
    history: History = env.data_manager.get_history("history")
    # ee_speed_mean, ee_speed_std = history.get_mean_and_std_dev(end_effector_speed_str)
    ee_position_b_mean, ee_position_b_std = history.get_mean_and_std_dev(ee_position_b_str)
    # # non_moving_mask = ee_speed_mean < 0.04
    non_moving_mask = (torch.sum(ee_position_b_std, dim=1) < 0.001) & (env.episode_length_buf > 50)
    return non_moving_mask


def canberry_dropped(
    env: HebiRLTaskEnv,
    canberry_cake_distance_key: str,
    canberry_eeframe_distance_key: str,
    chop_pose_key: str,
):
    canberry_cake_distance = env.data_manager.get_active_term("data", canberry_cake_distance_key)
    canberry_eeframe_distance = env.data_manager.get_active_term("data", canberry_eeframe_distance_key)
    chop_pose = env.data_manager.get_active_term("data", chop_pose_key)
    return (
        (canberry_cake_distance > 0.04)
        & (canberry_cake_distance < 0.15)
        & (chop_pose > -0.4)
        & (canberry_eeframe_distance > 0.05)
    ).view(-1)


def success_state(
    env: HebiRLTaskEnv,
    canberry_cake_distance_key: str,
    canberry_eeframe_distance_key: str,
    chop_pose_key: str,
):
    canberry_cake_distance = env.data_manager.get_active_term("data", canberry_cake_distance_key)
    canberry_eeframe_distance = env.data_manager.get_active_term("data", canberry_eeframe_distance_key)
    chop_pose = env.data_manager.get_active_term("data", chop_pose_key)

    return ((canberry_cake_distance < 0.022) & (chop_pose > -0.4) & (canberry_eeframe_distance > 0.05)).view(-1)


# def success_state(
#     env: HebiRLTaskEnv,
#     robot_name:str = "robot",
#     canberry_name: str = "canberry",
#     cake_name: str = "cake",
#     caneberry_offset: list[float] = [0.0, 0.0, 0.0],
#     cake_offset: list[float] = [0.0, 0.0, 0.0],
#     fixed_chop_frame_name = "frame_fixed_chop_tip",
#     free_chop_frame_name = "frame_free_chop_tip",

# ) -> torch.Tensor:
#     """Return ture if the RigidBody position reads nan
#     """
#     robot: Articulation = env.scene[robot_name]
#     canberry: RigidObject = env.scene[canberry_name]
#     cake: RigidObject = env.scene[cake_name]
#     canberry_offset_tensor = torch.tensor(caneberry_offset, device = env.device)
#     cake_offset_tensor = torch.tensor(cake_offset, device = env.device)
#     fixed_chop_frame = env.scene[fixed_chop_frame_name]
#     free_chop_frame = env.scene[free_chop_frame_name]

#     canberry_eeframe_distance = tycho_mdp.get_object_eeFrame_distance(canberry, canberry_offset_tensor, fixed_chop_frame, free_chop_frame)
#     chop_pose = robot.data.joint_pos[:, -1]
#     release_mask = chop_pose > -0.4
#     canberry_cake_distance = general_mdp.get_body1_body2_distance(canberry, cake, canberry_offset_tensor, cake_offset_tensor)
#     goal_reach_chop_release_mask = (canberry_cake_distance < 0.022) &\
#                                    (release_mask) &\
#                                    (canberry_eeframe_distance > 0.05)
#     return goal_reach_chop_release_mask
