# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
# import mdp as tycho_mdp
import omni.isaac.lab.envs.mdp as orbit_mdp
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from dataclasses import MISSING
##
# Pre-defined configs
##
episode_length = 15.0
from ... import mdp as track_goal_mdp
from lab.tycho.cfgs.scenes.empty_scene import SceneObjectSceneCfg, SceneCommandsCfg, SceneEventCfg, SceneRewardsCfg, SceneTerminationsCfg
import lab.tycho.cfgs.robots.hebi.robot_dynamics as rd
from lab.tycho.cfgs.robots.hebi.robot_cfg import FRAME_EE
from lab.tycho.cfgs.robots.hebi.robot_dynamics import RobotRewardsCfg


class ObjectSceneCfg():
    pass


@configclass
class IdealPDHebi_ObjectSceneCfg(SceneObjectSceneCfg, rd.RobotSceneCfg_IdealPD, ObjectSceneCfg):
    ee_frame = FRAME_EE


@configclass
class PwmMotorHebi_ObjectSceneCfg(SceneObjectSceneCfg, rd.RobotSceneCfg_HebiPwmMotor, ObjectSceneCfg):
    ee_frame = FRAME_EE


@configclass
class ImplicitMotorHebi_ObjectSceneCfg(SceneObjectSceneCfg, rd.RobotSceneCfg_ImplicityActuator, ObjectSceneCfg):
    ee_frame = FRAME_EE


@configclass
class ImplicitMotorOriginHebi_ObjectSceneCfg(SceneObjectSceneCfg, rd.RobotSceneCfg_ImplicityActuator_OriginHebi, ObjectSceneCfg):
    pass


@configclass
class ActionsCfg():
    """Action specifications for the MDP."""
    pass


@configclass
class ObservationsCfg:
    ""
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
class RewardsCfg(SceneRewardsCfg, RobotRewardsCfg):
    """Reward terms for the MDP."""
    pass
    ee_frame_goal_tracking = RewTerm(
        func=track_goal_mdp.ee_frame_goal_distance,
        params={"std": 0.1, "command_name": "ee_pose"},
        weight=3.0,)


@configclass
class DataCfg:
    # """Curriculum terms for the MDP."""
    pass


@configclass
class CommandsCfg(SceneCommandsCfg):
    """Command terms for the MDP."""

    ee_pose = orbit_mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="end_effector",
        resampling_time_range=(episode_length, episode_length),
        debug_vis=False,
        ranges=orbit_mdp.UniformPoseCommandCfg.Ranges(
            # CAUTION: to use the hebi's ik solver the roll needs to be (0.0, 0.0) for upright vector to be correct
            # however, for the isaac sim ik solver, the roll needs to be (1.57075, 1.57075)
            pos_x=(-0.45, -0.35), pos_y=(-0.4, -0.25), pos_z=(0.05, 0.14), roll=(1.57075, 1.57075), pitch=(3.14, 3.14), yaw=(0.5, 0.5)
        ),
    )


@configclass
class EventCfg(SceneEventCfg, rd.RobotRandomizationCfg):
    pass


@configclass
class TerminationsCfg(SceneTerminationsCfg):
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
    actions: ActionsCfg = MISSING
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    datas: DataCfg = DataCfg()

    def __post_init__(self):
        self.decimation = 10
        self.episode_length_s = episode_length
        # simulation settings
        self.sim.dt = 0.05 / self.decimation  # Agent: 20Hz, Motor: 500Hz
        self.sim.physx.min_position_iteration_count = 32
        self.sim.physx.min_velocity_iteration_count = 16
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625


class PwmMotorHebi_JointPos_GoalTracking_Env(IdealPDHebi_JointPos_GoalTracking_Env):
    def __post_init__(self):
        super().__post_init__()
        self.scene: ObjectSceneCfg = PwmMotorHebi_ObjectSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)


class ImplicitMotorHebi_JointPos_GoalTracking_Env(IdealPDHebi_JointPos_GoalTracking_Env):
    def __post_init__(self):
        super().__post_init__()
        self.scene: ObjectSceneCfg = ImplicitMotorHebi_ObjectSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)


@configclass
class EmptyRewardsCfg():
    """Reward terms for the MDP."""
    pass


class ImplicitMotorOriginHebi_JointPos_GoalTracking_Env(IdealPDHebi_JointPos_GoalTracking_Env):
    def __post_init__(self):
        super().__post_init__()
        self.scene: ObjectSceneCfg = ImplicitMotorOriginHebi_ObjectSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)
        self.rewards = EmptyRewardsCfg()
