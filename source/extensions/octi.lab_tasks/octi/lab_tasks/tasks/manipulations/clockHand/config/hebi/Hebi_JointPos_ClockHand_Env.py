# Copyright (c) 2022-2024, The Octi and Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from omni.isaac.lab.utils import configclass

# from omni.isaac.lab.envs import RLTaskEnvCfg
from octi.lab.envs import OctiManagerBasedRLEnvCfg

# import mdp as tycho_mdp
import omni.isaac.lab.envs.mdp as orbit_mdp
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm

from dataclasses import MISSING

##
# Pre-defined configs
##
from octi.lab_tasks.cfgs.scenes.clock_scene import (
    SceneObjectSceneCfg,
    SceneCommandsCfg,
    SceneEventCfg,
    SceneRewardsCfg,
    SceneTerminationsCfg,
)
import octi.lab_tasks.cfgs.robots.hebi.robot_dynamics as rd
from octi.lab_tasks.cfgs.robots.hebi.robot_cfg import FRAME_EE, CAMERA_WRIST, CAMERA_BASE  # noqa: F401


class ObjectSceneCfg:
    pass


@configclass
class IdealPDHebi_ObjectSceneCfg(SceneObjectSceneCfg, rd.RobotSceneCfg_IdealPD, ObjectSceneCfg):
    ee_frame = FRAME_EE


@configclass
class PwmMotorHebi_ObjectSceneCfg(SceneObjectSceneCfg, rd.RobotSceneCfg_HebiStrategy3Actuator, ObjectSceneCfg):
    ee_frame = FRAME_EE


@configclass
class ImplicitMotorHebi_ObjectSceneCfg(SceneObjectSceneCfg, rd.RobotSceneCfg_ImplicityActuator, ObjectSceneCfg):
    ee_frame = FRAME_EE
    # camera_wrist = CAMERA_WRIST
    # camera_base = CAMERA_BASE


@configclass
class DataCfg:
    # """Curriculum terms for the MDP."""
    pass


@configclass
class ActionsCfg:
    """Action will be defined at the moment class gets constructed"""

    pass


@configclass
class ObservationsCfg:
    """"""

    @configclass
    class PolicyCfg(ObsGroup, rd.RobotObservationsPolicyCfg):
        """Observations for policy group."""

        pass

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg(SceneRewardsCfg, rd.RobotRewardsCfg):
    """Reward terms for the MDP."""

    pass


@configclass
class CommandsCfg(SceneCommandsCfg):
    """Command terms for the MDP."""

    pass


@configclass
class EventCfg(SceneEventCfg, rd.RobotRandomizationCfg):
    pass


@configclass
class TerminationsCfg(SceneTerminationsCfg, rd.RobotTerminationsCfg):
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=orbit_mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


@configclass
class IdealPDHebi_JointPos_ClockHand_Env(OctiManagerBasedRLEnvCfg):
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
        self.decimation = 1
        self.episode_length_s = 12
        # simulation settings
        self.sim.dt = 0.02 / self.decimation  # Agent: 20Hz, Motor: 500Hz
        self.sim.physx.min_position_iteration_count = 32
        self.sim.physx.min_velocity_iteration_count = 16
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**22
        self.sim.physx.gpu_max_rigid_patch_count = 2**20


class PwmMotorHebi_JointPos_ClockHand_Env(IdealPDHebi_JointPos_ClockHand_Env):
    def __post_init__(self):
        super().__post_init__()
        self.scene: ObjectSceneCfg = PwmMotorHebi_ObjectSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)


class ImplicitMotorHebi_JointPos_ClockHand_Env(IdealPDHebi_JointPos_ClockHand_Env):
    def __post_init__(self):
        super().__post_init__()
        self.scene: ObjectSceneCfg = ImplicitMotorHebi_ObjectSceneCfg(
            num_envs=1, env_spacing=2.5, replicate_physics=False
        )
