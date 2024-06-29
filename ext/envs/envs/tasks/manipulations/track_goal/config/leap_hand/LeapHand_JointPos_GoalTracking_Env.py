# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg

# import mdp as tycho_mdp
import omni.isaac.lab.envs.mdp as orbit_mdp
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from dataclasses import MISSING


##
# Pre-defined configs
##
import ext.envs.cfgs.robots.leap_hand.robot_dynamics as rd
from ext.envs.cfgs.scenes.empty_scene import (
    SceneObjectSceneCfg,
    SceneCommandsCfg,
    SceneEventCfg,
    SceneRewardsCfg,
    SceneTerminationsCfg,
)

episode_length = 50.0


class ObjectSceneCfg:
    pass


@configclass
class ImplicitMotorLeap_ObjectSceneCfg(SceneObjectSceneCfg, rd.RobotSceneCfg_ImplicityActuator, ObjectSceneCfg):
    pass


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    pass


@configclass
class ObservationsCfg:
    """"""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=orbit_mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=orbit_mdp.joint_vel_rel)
        target_object_position = ObsTerm(func=orbit_mdp.generated_commands, params={"command_name": "ee_pose"})
        actions = ObsTerm(func=orbit_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg(SceneRewardsCfg, rd.RobotRewardsCfg):
    """Reward terms for the MDP."""

    # end_effector_position_tracking = RewTerm(
    #     func=mdp.position_command_error,
    #     weight=-2,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names="static_chop_tip"), "command_name": "ee_pose"},
    # )

    # end_effector_position_tracking_fine_grained = RewTerm(
    #     func=mdp.position_command_error_tanh,
    #     weight=4,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names="static_chop_tip"), "std": 0.1, "command_name": "ee_pose"},
    # )

    # end_effector_orientation_tracking = RewTerm(
    #     func=mdp.orientation_command_error,
    #     weight=-2,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names="static_chop_tip"), "command_name": "ee_pose"},
    # )


@configclass
class DataCfg:
    # """Curriculum terms for the MDP."""
    pass


@configclass
class CommandsCfg(SceneCommandsCfg):
    """Command terms for the MDP."""

    ee_pose = orbit_mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="palm_lower",
        resampling_time_range=(episode_length / 5, episode_length / 5),
        debug_vis=False,
        ranges=orbit_mdp.UniformPoseCommandCfg.Ranges(
            # CAUTION: to use the hebi's ik solver the roll needs to be (0.0, 0.0) for upright vector to be correct
            # however, for the isaac lab ik solver, the roll needs to be (1.57075, 1.57075)
            pos_x=(-0.45, -0.15),
            pos_y=(-0.5, -0.15),
            pos_z=(0.02, 0.3),
            roll=(1.57075, 1.57075),
            pitch=(3.14, 3.14),
            yaw=(0.0, 0.5),
        ),
    )


@configclass
class EventCfg(SceneEventCfg, rd.RobotRandomizationCfg):
    pass


@configclass
class TerminationsCfg(SceneTerminationsCfg, rd.RobotTerminationsCfg):
    """Termination terms for the MDP."""

    pass


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


@configclass
class ImplicitMotorLeap_JointPos_GoalTracking_Env(ManagerBasedRLEnvCfg):

    # Scene settings
    scene: ObjectSceneCfg = ImplicitMotorLeap_ObjectSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = MISSING  # type: ignore
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    datas: DataCfg = DataCfg()

    def __post_init__(self):
        self.decimation = 1
        self.episode_length_s = episode_length
        # simulation settings
        self.sim.dt = 0.02 / self.decimation  # Agent: 20Hz, Motor: 500Hz
        self.sim.physx.min_position_iteration_count = 16
        self.sim.physx.min_velocity_iteration_count = 8
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
