# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from omni.isaac.lab.utils import configclass

##
# Pre-defined configs
##
import octi.lab_assets.leap as leap
from ... import track_goal_env
episode_length = 50.0


@configclass
class TrackGoalLeap(track_goal_env.TrackGoalEnv):

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = leap.IMPLICIT_LEAP.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.commands.ee_pose.body_name="palm_lower"
        self.rewards.end_effector_position_tracking.params['asset_cfg'].body_names="palm_lower"
        self.rewards.end_effector_position_tracking.params['std'] = 2
        self.rewards.end_effector_position_tracking.weight *= 10
        self.rewards.end_effector_position_tracking_fine_grained.weight *= 10
        self.rewards.end_effector_position_tracking_fine_grained.params['asset_cfg'].body_names="palm_lower"
        self.rewards.end_effector_orientation_tracking.params['asset_cfg'].body_names="palm_lower"


@configclass
class TrackGoalLeapJointPosition(TrackGoalLeap):
    actions = leap.LeapJointPositionAction()


@configclass
class TrackGoalLeapMcIkAbs(TrackGoalLeap):
    actions = leap.LeapMcIkAbsoluteAction()


@configclass
class TrackGoalLeapMcIkDel(TrackGoalLeap):
    actions = leap.LeapMcIkDeltaAction()
