# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.managers import RewardTermCfg as RewTerm

# import mdp as tycho_mdp
import omni.isaac.lab.envs.mdp as orbit_mdp
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm  # noqa: F401
from ext.envs.cfgs.scenes.cube_scene import (
    SceneObjectSceneCfg,
    SceneCommandsCfg,
    SceneEventCfg,
    SceneRewardsCfg,
    SceneTerminationsCfg,
)
from octilab.envs import OctiManagerBasedRLEnvCfg
import ext.envs.envs.tasks.lift_cube.mdp as lift_cube_mdp

##
# Pre-defined configs
##

from ext.envs.cfgs.robots.hebi.robot_dynamics import (
    RobotSceneCfg_HebiStrategy3Actuator,  # noqa: F401
    RobotSceneCfg_ImplicityActuator,
    RobotSceneCfg_IdealPD,
)

from ext.envs.cfgs.robots.hebi.robot_dynamics import (
    RobotActionsCfg_HebiJointPosition,  # noqa: F401
    RobotActionsCfg_HebiIkAbsoluteDls,
    RobotActionsCfg_HebiIkDeltaDls,
)

from ext.envs.cfgs.robots.hebi.robot_dynamics import RobotObservationsPolicyCfg
from ext.envs.cfgs.robots.hebi.robot_dynamics import RobotRewardsCfg
from ext.envs.cfgs.robots.hebi.robot_dynamics import RobotRandomizationCfg
import ext.envs.cfgs.robots.hebi.robot_dynamics as rd
from ext.envs.cfgs.robots.hebi.robot_cfg import FRAME_EE, CAMERA_WRIST, CAMERA_BASE  # noqa: F401


@configclass
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
class ActionsCfg(RobotActionsCfg_HebiJointPosition):
    """Action specifications for the MDP."""

    pass


@configclass
class DataCfg:
    # """Curriculum terms for the MDP."""
    pass


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup, RobotObservationsPolicyCfg):
        """Observations for policy group."""

        object_position = ObsTerm(func=lift_cube_mdp.object_position_in_robot_root_frame)
        # object_frame_position = ObsTerm(func=lift_cube_mdp.object_frame_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=orbit_mdp.generated_commands, params={"command_name": "object_pose"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg(SceneRewardsCfg, RobotRewardsCfg):
    """Reward terms for the MDP."""

    reward_object_ee_distance = RewTerm(
        func=lift_cube_mdp.reward_object_ee_distance,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "fixed_chop_frame_cfg": SceneEntityCfg("frame_free_chop_tip"),
            "free_chop_frame_cfg": SceneEntityCfg("frame_free_chop_tip"),
        },
        weight=1.0,
    )
    # lifting_object = RewTerm(func=lift_cube_mdp.object_is_lifted, params={"minimal_height": 0.1}, weight=15.0)

    # object_goal_tracking = RewTerm(
    #     func=lift_cube_mdp.object_goal_distance,
    #     params={"std": 0.3, "minimal_height": 0.1, "command_name": "object_pose"},
    #     weight=30.0,
    # )

    # object_goal_tracking_fine_grained = RewTerm(
    #     func=lift_cube_mdp.object_goal_distance,
    #     params={"std": 0.05, "minimal_height": 0.1, "command_name": "object_pose"},
    #     weight=10.0,
    # )


@configclass
class CommandsCfg(SceneCommandsCfg):
    """Command terms for the MDP."""

    object_pose = orbit_mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="end_effector",
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=orbit_mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-0.7, -0.5), pos_y=(-0.25, 0.25), pos_z=(0.1, 0.2), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )


@configclass
class EventCfg(SceneEventCfg, RobotRandomizationCfg):
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
class IdealPDHebi_JointPos_LiftCube_Env(OctiManagerBasedRLEnvCfg):

    scene: ObjectSceneCfg = IdealPDHebi_ObjectSceneCfg(num_envs=2048, env_spacing=2.5, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    datas: DataCfg = DataCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 10
        self.episode_length_s = 5.0
        # simulation settings
        self.sim.dt = 0.05 / self.decimation
        self.sim.physx.min_position_iteration_count = 1
        self.sim.physx.min_velocity_iteration_count = 0
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625


class PwmMotorHebi_JointPos_LiftCube_Env(IdealPDHebi_JointPos_LiftCube_Env):
    def __post_init__(self):
        super().__post_init__()
        self.scene: ObjectSceneCfg = PwmMotorHebi_ObjectSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)


class ImplicitMotorHebi_JointPos_LiftCube_Env(IdealPDHebi_JointPos_LiftCube_Env):
    def __post_init__(self):
        super().__post_init__()
        self.scene: ObjectSceneCfg = ImplicitMotorHebi_ObjectSceneCfg(
            num_envs=1, env_spacing=2.5, replicate_physics=False
        )
