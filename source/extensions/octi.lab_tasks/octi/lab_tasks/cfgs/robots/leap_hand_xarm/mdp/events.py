from __future__ import annotations
from typing import TYPE_CHECKING, Sequence
import torch
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg
import omni.isaac.lab.utils.math as math_utils
if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv


def update_joint_target_positions_to_current(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_name: str
):
    asset: Articulation = env.scene[asset_name]
    joint_pos_target = asset.data.joint_pos
    asset.set_joint_position_target(joint_pos_target)


def reset_robot_to_default(env: ManagerBasedEnv, env_ids: torch.Tensor, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Reset the scene to the default state specified in the scene configuration."""
    robot: Articulation = env.scene[robot_cfg.name]
    default_root_state = robot.data.default_root_state[env_ids].clone()
    default_root_state[:, 0:3] += env.scene.env_origins[env_ids]
    # set into the physics simulation
    robot.write_root_state_to_sim(default_root_state, env_ids=env_ids)
    # obtain default joint positions
    default_joint_pos = robot.data.default_joint_pos[env_ids].clone()
    default_joint_vel = robot.data.default_joint_vel[env_ids].clone()
    # set into the physics simulation
    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)


def reset_joints_by_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    joint_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints with offsets around the default position and velocity by the given ranges.

    This function samples random values from the given ranges and biases the default joint positions and velocities
    by these values. The biased values are then set into the physics simulation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # get default joint state
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()

    # bias these values randomly
    joint_pos[:, joint_ids] += math_utils.sample_uniform(*position_range, joint_pos.shape, joint_pos.device)[:, joint_ids]
    joint_vel += math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)

    # clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    # clamp joint vel to limits
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)