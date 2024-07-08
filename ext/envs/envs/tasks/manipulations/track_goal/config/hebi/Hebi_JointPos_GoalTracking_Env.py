# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import SceneEntityCfg

# import mdp as tycho_mdp
import omni.isaac.lab.envs.mdp as lab_mdp
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from dataclasses import MISSING

##
# Pre-defined configs
##
import omni.isaac.lab_tasks.manager_based.manipulation.reach.mdp as mdp
from ext.envs.cfgs.scenes.empty_scene import (
    SceneObjectSceneCfg,
    SceneCommandsCfg,
    SceneEventCfg,
    SceneRewardsCfg,
    SceneTerminationsCfg,
)
import ext.envs.cfgs.robots.hebi.robot_dynamics as rd
from ext.envs.cfgs.robots.hebi.robot_cfg import FRAME_EE
from ext.envs.cfgs.robots.hebi.robot_dynamics import RobotRewardsCfg
import ext.envs.cfgs.robots.hebi.mdp as hebimdp

episode_length = 50.0


class ObjectSceneCfg:
    pass


@configclass
class IdealPDHebi_ObjectSceneCfg(SceneObjectSceneCfg, rd.RobotSceneCfg_IdealPD, ObjectSceneCfg):
    ee_frame = FRAME_EE


@configclass
class Strategy3Hebi_ObjectSceneCfg(SceneObjectSceneCfg, rd.RobotSceneCfg_HebiStrategy3Actuator, ObjectSceneCfg):
    ee_frame = FRAME_EE


@configclass
class Strategy4Hebi_ObjectSceneCfg(SceneObjectSceneCfg, rd.RobotSceneCfg_HebiStrategy4Actuator, ObjectSceneCfg):
    ee_frame = FRAME_EE


@configclass
class ImplicitMotorHebi_ObjectSceneCfg(SceneObjectSceneCfg, rd.RobotSceneCfg_ImplicityActuator, ObjectSceneCfg):
    ee_frame = FRAME_EE


@configclass
class EffortHebi_ObjectSceneCfg(SceneObjectSceneCfg, rd.RobotSceneCfg_HebiEffort, ObjectSceneCfg):
    ee_frame = FRAME_EE


@configclass
class HebiDCMotorHebi_ObjectSceneCfg(SceneObjectSceneCfg, rd.RobotSceneCfg_HebiDCMotor, ObjectSceneCfg):
    ee_frame = FRAME_EE


@configclass
class DCMotorHebi_ObjectSceneCfg(SceneObjectSceneCfg, rd.RobotSceneCfg_DCMotor, ObjectSceneCfg):
    ee_frame = FRAME_EE


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

        joint_pos = ObsTerm(func=lab_mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=lab_mdp.joint_vel_rel)
        target_object_position = ObsTerm(func=lab_mdp.generated_commands, params={"command_name": "ee_pose"})
        actions = ObsTerm(func=lab_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg(SceneRewardsCfg, RobotRewardsCfg):
    """Reward terms for the MDP."""

    end_effector_position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="static_chop_tip"),
            "std": 0.5,
            "command_name": "ee_pose",
        },
    )

    end_effector_position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=4,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="static_chop_tip"),
            "std": 0.1,
            "command_name": "ee_pose",
        },
    )

    end_effector_orientation_tracking = RewTerm(
        func=hebimdp.orientation_command_error_tanh,
        weight=2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="static_chop_tip"),
            "std": 0.5,
            "command_name": "ee_pose"},
    )


@configclass
class DataCfg:
    # """Curriculum terms for the MDP."""
    pass


@configclass
class CommandsCfg(SceneCommandsCfg):
    """Command terms for the MDP."""

    ee_pose = lab_mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="static_chop_tip",
        resampling_time_range=(episode_length / 5, episode_length / 5),
        debug_vis=False,
        ranges=lab_mdp.UniformPoseCommandCfg.Ranges(
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
class IdealPDHebi_JointPos_GoalTracking_Env(ManagerBasedRLEnvCfg):

    # Scene settings
    scene: ObjectSceneCfg = IdealPDHebi_ObjectSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)
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
        self.decimation = 2
        self.episode_length_s = episode_length
        # simulation settings
        self.sim.dt = 0.04 / self.decimation  # Env_Hz = Agent_Hz / decimation**2
        self.sim.physx.min_position_iteration_count = 16
        self.sim.physx.min_velocity_iteration_count = 8
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 24 * 1024
        self.sim.physx.gpu_max_rigid_patch_count = 5 * 2 ** 17
        self.sim.physx.friction_correlation_distance = 0.00625


class Strategy3MotorHebi_JointPos_GoalTracking_Env(IdealPDHebi_JointPos_GoalTracking_Env):
    def __post_init__(self):
        super().__post_init__()
        self.scene: ObjectSceneCfg = Strategy3Hebi_ObjectSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)


class Strategy4MotorHebi_JointPos_GoalTracking_Env(IdealPDHebi_JointPos_GoalTracking_Env):
    def __post_init__(self):
        super().__post_init__()
        self.scene: ObjectSceneCfg = Strategy4Hebi_ObjectSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)


class ImplicitMotorHebi_JointPos_GoalTracking_Env(IdealPDHebi_JointPos_GoalTracking_Env):
    def __post_init__(self):
        super().__post_init__()
        self.scene: ObjectSceneCfg = ImplicitMotorHebi_ObjectSceneCfg(
            num_envs=1, env_spacing=2.5, replicate_physics=False
        )


class EffortHebi_JointPos_GoalTracking_Env(IdealPDHebi_JointPos_GoalTracking_Env):
    def __post_init__(self):
        super().__post_init__()
        self.scene: ObjectSceneCfg = EffortHebi_ObjectSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)


class HebiDCMotorHebi_JointPos_GoalTracking_Env(IdealPDHebi_JointPos_GoalTracking_Env):
    def __post_init__(self):
        super().__post_init__()
        self.scene: ObjectSceneCfg = HebiDCMotorHebi_ObjectSceneCfg(
            num_envs=1, env_spacing=2.5, replicate_physics=False
        )


class DCMotorHebi_JointPos_GoalTracking_Env(IdealPDHebi_JointPos_GoalTracking_Env):
    def __post_init__(self):
        super().__post_init__()
        self.scene: ObjectSceneCfg = DCMotorHebi_ObjectSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)


class IdealPDHebi_JointPos_GoalTracking_Decimate1_Env(IdealPDHebi_JointPos_GoalTracking_Env):
    def __post_init__(self):
        super().__post_init__()
        agent_update_hz = self.decimation * self.sim.dt
        self.decimation = 1
        self.sim.dt = agent_update_hz / self.decimation


class IdealPDHebi_JointPos_GoalTracking_Decimate2_Env(IdealPDHebi_JointPos_GoalTracking_Env):
    def __post_init__(self):
        super().__post_init__()
        agent_update_hz = self.decimation * self.sim.dt
        self.decimation = 2
        self.sim.dt = agent_update_hz / self.decimation


class IdealPDHebi_JointPos_GoalTracking_Decimate5_Env(IdealPDHebi_JointPos_GoalTracking_Env):
    def __post_init__(self):
        super().__post_init__()
        agent_update_hz = self.decimation * self.sim.dt
        self.decimation = 5
        self.sim.dt = agent_update_hz / self.decimation
