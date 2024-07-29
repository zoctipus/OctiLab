# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass
from ... import fetching_env
# import octi.lab.envs.mdp as octi_mdp
import math
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise
##
# Pre-defined configs
##
from ... import fetching_env
from octi.lab_assets.unitree import UNITREE_A1_IMPLICIT_ACTUATOR_CFG
import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)


@configclass
class UnitreeA1FetchingRoughCfg(fetching_env.LocomotionFetchingRoughEnvCfg):
    actions: ActionsCfg = ActionsCfg()
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = UNITREE_A1_IMPLICIT_ACTUATOR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/trunk"
        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.size = (6.0, 6.0)
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].step_height_range = (0.025, 0.165)
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].step_height_range = (0.025, 0.165)
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].platform_width = 1.5
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].platform_width = 1.5

        # event
        self.events.push_robot = None  # type: ignore
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "trunk"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "trunk"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        }

        # rewards
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        # self.rewards.feet_air_time.weight = 5
        self.rewards.undesired_contacts = None
        # self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.dof_acc_l2.weight = -2.5e-7

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "trunk"

