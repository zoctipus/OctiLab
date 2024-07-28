# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm


# Task Composition Import
from ... import track_goal_env
import octi.lab_tasks.tasks.manipulation.track_goal.mdp as mdp
import octi.lab_assets.tycho.mdp as tycho_mdp
import octi.lab_assets.tycho as tycho


class SceneCfg(track_goal_env.SceneCfg):
    ee_frame = tycho.FRAME_EE
    robot = tycho.HEBI_IMPLICITY_ACTUATOR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    frame_fixed_chop_tip = tycho.FRAME_FIXED_CHOP_TIP
    frame_free_chop_tip = tycho.FRAME_FREE_CHOP_TIP


@configclass
class RewardsCfg(track_goal_env.RewardsCfg):
    """Reward terms for the MDP."""
    # action penalty
    velocity_punishment = RewTerm(func=mdp.joint_vel_limits, weight=-1, params={"soft_ratio": 1})

    punish_hand_tilted = RewTerm(
        func=tycho_mdp.punish_hand_tilted,
        weight=1,
    )

    punish_touching_ground = RewTerm(
        func=tycho_mdp.punish_touching_ground,
        weight=0.3,
    )

    punish_bad_elbow_posture = RewTerm(
        func=tycho_mdp.punish_bad_elbow_shoulder_posture,
        weight=1,
    )



@configclass
class TerminationsCfg(track_goal_env.TerminationsCfg):
    """Termination terms for the MDP."""
    robot_invalid_state = DoneTerm(func=mdp.invalid_state, params={"asset_cfg": SceneEntityCfg("robot")})

    robot_extremely_bad_posture = DoneTerm(
        func=tycho_mdp.terminate_extremely_bad_posture,
        params={"probability": 0.5, "robot_cfg": SceneEntityCfg("robot")},
    )


@configclass
class GoalTrackingTychoEnv(track_goal_env.TrackGoalEnv):

    # Scene settings
    scene: SceneCfg = SceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.ee_pose.body_name="static_chop_tip"
        self.commands.ee_pose.ranges.pos_x = (-0.45, -0.1)
        self.rewards.end_effector_position_tracking.params['asset_cfg'].body_names="static_chop_tip"
        self.rewards.end_effector_position_tracking_fine_grained.params['asset_cfg'].body_names="static_chop_tip"
        self.rewards.end_effector_orientation_tracking.params['asset_cfg'].body_names="static_chop_tip"


@configclass
class GoalTrackingTychoIkdelta(GoalTrackingTychoEnv):
    actions = tycho.IkdeltaAction()


@configclass
class GoalTrackingTychoIkabsolute(GoalTrackingTychoEnv):
    actions = tycho.IkabsoluteAction()


@configclass
class GoalTrackingTychoJointPosition(GoalTrackingTychoEnv):
    actions = tycho.JointPositionAction()


@configclass
class GoalTrackingTychoJointEffort(GoalTrackingTychoEnv):
    actions = tycho.JointEffortAction()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = tycho.HEBI_EFFORT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
