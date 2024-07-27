from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject, Articulation
from omni.isaac.lab.managers import SceneEntityCfg
if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def reward_cross_finger_similarity(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot: Articulation = env.scene[robot_cfg.name]
    # j12, j1, j5, j9
    joint_knuckles = robot.data.joint_pos[:, [6, 5, 7, 8]]
    # j14, j2, j6, j10
    joint_pips = robot.data.joint_pos[:, [14, 13, 15, 16]]
    # j15, j3, j7, j11,
    joint_dips = robot.data.joint_pos[:, [18, 17, 19, 20]]
    # Calculate the differences between corresponding joints
    weights = torch.tensor([0.5, 0.7, 1.0], device=joint_knuckles.device)
    knuckle_diff = (joint_knuckles[:, 1:] - joint_knuckles[:, :-1]) * weights
    pip_diff = (joint_pips[:, 1:] - joint_pips[:, :-1]) * weights
    dip_diff = (joint_dips[:, 1:] - joint_dips[:, :-1]) * weights
    # Calculate the reward based on the differences
    knuckles_diff = torch.mean(torch.abs(knuckle_diff), dim=1)
    pip_diff = torch.mean(torch.abs(pip_diff), dim=1)
    dip_diff = torch.mean(torch.abs(dip_diff), dim=1)
    # Combine the rewards
    total_diff = (pip_diff + knuckles_diff + dip_diff) / 3
    # Smaller differences result in higher rewards, but max at 1
    reward = 1 - torch.tanh(total_diff / 0.5)
    return reward


def reward_intra_finger_similarity(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot: Articulation = env.scene[robot_cfg.name]

    # Extract joint positions for each finger separately
    # j12, j15, j3
    finger_1_joints = robot.data.joint_pos[:, [6, 14, 18]]
    # j1, j2, j7
    finger_2_joints = robot.data.joint_pos[:, [5, 13, 17]]
    # j5, j6, j11
    finger_3_joints = robot.data.joint_pos[:, [7, 15, 19]]
    # j9, j10, j12
    finger_4_joints = robot.data.joint_pos[:, [8, 16, 20]]
    # Calculate the differences between the joints of each finger
    weights = torch.tensor([0.5, 1.0], device=finger_2_joints.device)
    finger_1_diff = (finger_1_joints[:, 1:] - finger_1_joints[:, :-1]) * weights
    finger_2_diff = (finger_2_joints[:, 1:] - finger_2_joints[:, :-1]) * weights
    finger_3_diff = (finger_3_joints[:, 1:] - finger_3_joints[:, :-1]) * weights
    finger_4_diff = (finger_4_joints[:, 1:] - finger_4_joints[:, :-1]) * weights

    # Calculate the mean absolute differences for each finger
    finger_1_mean_diff = torch.mean(torch.abs(finger_1_diff), dim=1)
    finger_2_mean_diff = torch.mean(torch.abs(finger_2_diff), dim=1)
    finger_3_mean_diff = torch.mean(torch.abs(finger_3_diff), dim=1)
    finger_4_mean_diff = torch.mean(torch.abs(finger_4_diff), dim=1)

    # Combine the mean differences
    total_mean_diff = (finger_1_mean_diff + finger_2_mean_diff + finger_3_mean_diff + finger_4_mean_diff) / 4

    # Calculate the reward: smaller differences result in higher rewards
    reward = 1 - torch.tanh(total_mean_diff / 0.5)

    return reward


def reward_fingers_object_distance(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    tips_pos = robot.data.body_pos_w[..., 25:28, :]
    dips_pos = robot.data.body_pos_w[..., 17:20, :]
    object_pos = object.data.body_pos_w

    dips_distance = torch.mean(torch.norm(dips_pos - object_pos, dim=1))
    tips_distance = torch.mean(torch.norm(tips_pos - object_pos, dim=1))

    distance = (dips_distance + tips_distance) / 2
    reward = 1 - torch.tanh(distance / 0.2)

    return reward
