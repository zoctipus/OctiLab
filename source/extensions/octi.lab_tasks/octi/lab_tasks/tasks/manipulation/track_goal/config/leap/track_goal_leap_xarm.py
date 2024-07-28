# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from ... import track_goal_env
import octi.lab_assets.leap as leap

##
# Pre-defined configs
##
import omni.isaac.lab_tasks.manager_based.manipulation.reach.mdp as mdp
import octi.lab_assets.leap.mdp as leap_mdp
import octi.lab_assets.leap as leap
from ... import track_goal_env
##
# Pre-defined configs
##
@configclass
class EventCfg:
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
class TrackGoalLeapXarm(track_goal_env.TrackGoalEnv):
    events: EventCfg = EventCfg()
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = leap.IMPLICIT_LEAP_XARM.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.commands.ee_pose.body_name="palm_lower"
        self.rewards.end_effector_position_tracking.params['asset_cfg'].body_names="palm_lower"
        self.rewards.end_effector_position_tracking_fine_grained.params['asset_cfg'].body_names="palm_lower"
        self.rewards.end_effector_orientation_tracking.params['asset_cfg'].body_names="palm_lower"


@configclass
class TrackGoalLeapXarmJointPosition(TrackGoalLeapXarm):
    actions = leap.LeapXarmJointPositionAction()


@configclass
class TrackGoalLeapXarmMcIkAbs(TrackGoalLeapXarm):
    actions = leap.LeapXarmMcIkAbsoluteAction()


@configclass
class TrackGoalLeapXarmMcIkDel(TrackGoalLeapXarm):
    actions = leap.LeapXarmMcIkDeltaAction()
