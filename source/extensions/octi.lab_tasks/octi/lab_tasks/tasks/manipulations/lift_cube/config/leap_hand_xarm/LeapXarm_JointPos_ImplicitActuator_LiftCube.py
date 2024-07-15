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
from octi.lab_tasks.cfgs.scenes.cube_scene import (
    SceneObjectSceneCfg,
    SceneCommandsCfg,
    SceneEventCfg,
    SceneRewardsCfg,
    SceneTerminationsCfg,
)
from dataclasses import MISSING
from octi.lab.envs import OctiManagerBasedRLEnvCfg
import octi.lab_tasks.cfgs.robots.leap_hand_xarm.mdp as leap_hand_xarm_mdp
import octi.lab_tasks.tasks.manipulations.lift_cube.mdp as lift_cube_mdp
import octi.lab.envs.mdp as octilab_mdp
##
# Pre-defined configs
##

from octi.lab_tasks.cfgs.robots.leap_hand_xarm.robot_dynamics import (   # noqa: F401
    RobotActionsCfg_JointPosition,
)

from octi.lab_tasks.cfgs.robots.leap_hand_xarm.robot_dynamics import RobotObservationsPolicyCfg
from octi.lab_tasks.cfgs.robots.leap_hand_xarm.robot_dynamics import RobotRewardsCfg
from octi.lab_tasks.cfgs.robots.leap_hand_xarm.robot_dynamics import RobotRandomizationCfg
import octi.lab_tasks.cfgs.robots.leap_hand_xarm.robot_dynamics as rd
from octi.lab_tasks.cfgs.robots.leap_hand_xarm.robot_cfg import FRAME_EE


@configclass
class ObjectSceneCfg:
    pass


@configclass
class ImplicitMotorHebi_ObjectSceneCfg(SceneObjectSceneCfg, rd.RobotSceneCfg_ImplicityActuator, ObjectSceneCfg):
    ee_frame = FRAME_EE


@configclass
class ActionsCfg():
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
        func=octilab_mdp.reward_body1_frame2_distance,
        params={
            "body_cfg": SceneEntityCfg("object"),
            "frame_cfg": SceneEntityCfg("ee_frame"),
        },
        weight=3.0,
    )

    reward_fingers_object_distance = RewTerm(
        func=leap_hand_xarm_mdp.reward_fingers_object_distance,
        params={"object_cfg": SceneEntityCfg("object")},
        weight=10.0,
    )

    lifting_object = RewTerm(func=lift_cube_mdp.object_is_lifted,
                             params={"minimal_height": 0.07, "object_cfg": SceneEntityCfg("object")},
                             weight=40.0)

    object_goal_tracking = RewTerm(
        func=lift_cube_mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.07, "command_name": "object_pose"},
        weight=40.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=lift_cube_mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.07, "command_name": "object_pose"},
        weight=80.0,
    )


@configclass
class CommandsCfg(SceneCommandsCfg):
    """Command terms for the MDP."""

    object_pose = orbit_mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="palm_lower",
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=orbit_mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.2, 0.5), pos_y=(-0.35, 0.35), pos_z=(0.15, 0.3), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
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
class ImplicitMotorLeapXarm_JointPos_LiftCube_Env(OctiManagerBasedRLEnvCfg):

    scene: ObjectSceneCfg = ImplicitMotorHebi_ObjectSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    datas: DataCfg = DataCfg()
    actions: ActionsCfg = MISSING
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 1
        self.episode_length_s = 4.0
        # simulation settings
        self.sim.dt = 0.02 / self.decimation
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.gpu_max_rigid_patch_count = 5 * 2 ** 16
