from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from omni.isaac.lab.assets import Articulation, RigidObject
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


def dump_data(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Push the asset by setting the root velocity to a random value within the given ranges.

    This creates an effect similar to pushing the asset with a random impulse that changes the asset's velocity.
    It samples the root velocity from the given ranges and sets the velocity into the physics simulation.

    The function takes a dictionary of velocity ranges for each axis and rotation. The keys of the dictionary
    are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form ``(min, max)``.
    If the dictionary does not contain a key, the velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # positions
    pos_w = asset.data.root_pos_w[env_ids]
    if len(pos_w) == env.num_envs:
        env.logger.append_to_buffer(pos_w)


def reset_below_min_height(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: RigidObject = env.scene[asset_cfg.name]
    reset_id = asset.data.root_pos_w[:, 2][env_ids] < minimum_height
    if any(reset_id):
        reset_root_state_uniform(
            env=env,
            env_ids=env_ids[reset_id],
            pose_range=pose_range,
            velocity_range=velocity_range,
            asset_cfg=asset_cfg,
        )


def reset_upon_close(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    reset_distance: float,
    compare_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reset_asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    # extract the used quantities (to enable type-hinting)
    compare_asset: RigidObject | Articulation = env.scene[compare_asset_cfg.name]
    reset_asset: RigidObject | Articulation = env.scene[reset_asset_cfg.name]
    # get default root state
    reset_asset_w = reset_asset.data.root_pos_w[env_ids]
    compare_asset_w = compare_asset.data.root_pos_w[env_ids]
    env_ids = env_ids[torch.norm(reset_asset_w - compare_asset_w, dim=1) < reset_distance]
    if len(env_ids) > 0:
        root_states = reset_asset.data.default_root_state[env_ids].clone()
        # poses
        range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=reset_asset.device)
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=reset_asset.device)

        positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
        orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
        # velocities
        range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=reset_asset.device)
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=reset_asset.device)

        velocities = root_states[:, 7:13] + rand_samples

        # set into the physics simulation
        reset_asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
        reset_asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)

        # if "goal_reached_count" not in env.extensions:
        #     env.extensions["goal_reached_count"] = torch.zeros((env.num_envs,), dtype=torch.long, device=env.device)

        env.extensions["goal_reached_count"][env_ids] += 1


def reset_root_state_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_states[:, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
