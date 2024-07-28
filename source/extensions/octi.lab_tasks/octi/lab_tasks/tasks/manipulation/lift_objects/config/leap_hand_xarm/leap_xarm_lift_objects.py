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
from dataclasses import MISSING
from octi.lab.envs import OctiManagerBasedRLEnvCfg
# import octi.lab_tasks.cfgs.robots.leap_hand_xarm.mdp as leap_hand_xarm_mdp
import octi.lab_tasks.tasks.manipulation.lift_cube.mdp as task_mdp
import octi.lab_assets.leap as leap
import octi.lab_assets.leap.mdp as leap_mdp

##
# Pre-defined configs
##
# from octi.lab_tasks.cfgs.robots.leap_hand_xarm.robot_cfg import FRAME_EE
import octi.lab_assets.leap as leap
from ... import lift_objects


@configclass
class ObjectSceneCfg(lift_objects.ObjectSceneCfg):
    robot = leap.IMPLICIT_LEAP_XARM.replace(prim_path="{ENV_REGEX_NS}/Robot")
    ee_frame = leap.FRAME_EE


@configclass
class ActionsCfg:
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
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        robot_actions = ObsTerm(func=task_mdp.last_action)
        robot_joint_pos = ObsTerm(func=task_mdp.joint_pos_rel)
        robot_joint_vel = ObsTerm(func=task_mdp.joint_vel_rel)
        object_position = ObsTerm(func=task_mdp.object_position_in_robot_root_frame)
        # object_frame_position = ObsTerm(func=task_mdp.object_frame_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=task_mdp.generated_commands, params={"command_name": "object_pose"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg(lift_objects.RewardsCfg):
    """Reward terms for the MDP."""
    action_rate = RewTerm(func=task_mdp.action_rate_l2, weight=-0.01)

    joint_vel = RewTerm(
        func=task_mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    reward_object_ee_distance = RewTerm(
        func=task_mdp.reward_body1_frame2_distance,
        params={
            "body_cfg": SceneEntityCfg("object"),
            "frame_cfg": SceneEntityCfg("ee_frame"),
        },
        weight=3.0,
    )

    reward_fingers_object_distance = RewTerm(
        func=leap_mdp.reward_fingers_object_distance,
        params={"object_cfg": SceneEntityCfg("object")},
        weight=10.0,
    )

    object_goal_tracking = RewTerm(
        func=task_mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.082, "command_name": "object_pose"},
        weight=40.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=task_mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.082, "command_name": "object_pose"},
        weight=80.0,
    )


@configclass
class CommandsCfg:
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
class EventCfg(lift_objects.EventCfg):
    reset_robot_joint = EventTerm(
        func=leap_mdp.reset_joints_by_offset,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": [0.0, 1.5],
            "velocity_range": [-0.1, 0.1],
            "joint_ids": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        },
        mode="reset",
    )


@configclass
class TerminationsCfg(lift_objects.TerminationsCfg):
    """Termination terms for the MDP."""
    pass


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    pass


@configclass
class LiftObejctsLeapXarmEnv(OctiManagerBasedRLEnvCfg):

    scene: ObjectSceneCfg = ObjectSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    datas: DataCfg = DataCfg()
    actions: ActionsCfg = MISSING  # type: ignore
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


@configclass
class ObejctsLiftLeapXarmJointPosition(LiftObejctsLeapXarmEnv):
    actions = leap.LeapXarmJointPositionAction()


@configclass
class ObejctsLiftLeapXarmMcIkAbs(LiftObejctsLeapXarmEnv):
    actions = leap.LeapXarmMcIkabsoluteAction()
