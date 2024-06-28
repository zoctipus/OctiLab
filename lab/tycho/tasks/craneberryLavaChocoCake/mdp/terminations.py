from __future__ import annotations

import torch
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from octilab.envs import OctiManagerBasedRLEnv
from octilab.managers.data_manager import History
"""
MDP terminations.
"""


def chop_missing(
    env: OctiManagerBasedRLEnv,
    canberry_eeframe_distance_key: str,
    chop_pose_key: str,
):
    canberry_eeframe_distance = env.data_manager.get_active_term("data", canberry_eeframe_distance_key)
    chop_pose = env.data_manager.get_active_term("data", chop_pose_key)
    return ((chop_pose < -0.57) & (canberry_eeframe_distance < 0.035)).view(-1)


def non_moving_abnormalty(
    env: OctiManagerBasedRLEnv,
    end_effector_speed_str: str = "end_effector_speed",
    ee_position_b_str: str = "ee_position_b"

):
    history: History = env.data_manager.get_history("history")
    # ee_speed_mean, ee_speed_std = history.get_mean_and_std_dev(end_effector_speed_str)
    ee_position_b_mean, ee_position_b_std = history.get_mean_and_std_dev(ee_position_b_str)
    # # non_moving_mask = ee_speed_mean < 0.04
    non_moving_mask = (torch.sum(ee_position_b_std, dim=1) < 0.001) & (env.episode_length_buf > 50)
    return non_moving_mask


def canberry_dropped(
    env: OctiManagerBasedRLEnv,
    canberry_cake_distance_key : str,
    canberry_eeframe_distance_key: str,
    chop_pose_key: str,
):
    canberry_cake_distance = env.data_manager.get_active_term("data", canberry_cake_distance_key)
    canberry_eeframe_distance = env.data_manager.get_active_term("data", canberry_eeframe_distance_key)
    chop_pose = env.data_manager.get_active_term("data", chop_pose_key)
    return ((canberry_cake_distance > 0.04) &\
            (canberry_cake_distance < 0.15) &\
            (chop_pose > -0.4) &\
            (canberry_eeframe_distance > 0.05)).view(-1)
    

def success_state(
    env: OctiManagerBasedRLEnv,
    canberry_cake_distance_key : str,
    canberry_eeframe_distance_key: str,
    chop_pose_key: str,
):
    canberry_cake_distance = env.data_manager.get_active_term("data", canberry_cake_distance_key)
    canberry_eeframe_distance = env.data_manager.get_active_term("data", canberry_eeframe_distance_key)
    chop_pose = env.data_manager.get_active_term("data", chop_pose_key)

    return ((canberry_cake_distance < 0.022) & (chop_pose > -0.4) & (canberry_eeframe_distance > 0.05)).view(-1)
