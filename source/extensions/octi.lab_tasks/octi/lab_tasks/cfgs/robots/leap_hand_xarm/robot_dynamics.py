from __future__ import annotations

from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
import omni.isaac.lab.envs.mdp as lab_mdp
from . import mdp as leap_xarm_mdp

"""
ROBOT ACTION CFGS
"""
from omni.isaac.lab.envs.mdp.actions.actions_cfg import (
    JointPositionActionCfg,
    JointEffortActionCfg,
)
from octi.lab.envs.mdp.actions.actions_cfg import MultiConstraintsDifferentialInverseKinematicsActionCfg
from omni.isaac.lab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from octi.lab.controllers.differential_ik_cfg import MultiConstraintDifferentialIKControllerCfg
from . import IMPLICIT_LEAP_XARM  # noqa: F401


@configclass
class RobotSceneCfg_ImplicityActuator:
    """
    HEBI Implicity Actuator is a simple low PD that groups "X8_9", "X8_16", "x5" typed
    motor into identical PD values.
    """

    robot = IMPLICIT_LEAP_XARM.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class RobotActionsCfg_JointPosition:
    """Action specifications for the MDP."""

    joint_position: JointPositionActionCfg = JointPositionActionCfg(asset_name="robot", joint_names=["j.*", "a.*"], scale=1)


@configclass
class RobotActionsCfg_JointEffort:
    """Action specifications for the MDP."""

    joint_efforts: JointEffortActionCfg = JointEffortActionCfg(asset_name="robot", joint_names=["j.*", "a.*"], scale=0.1)


@configclass
class RobotActionsCfg_MCIkAbsoluteDls:
    index_finger = MultiConstraintsDifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["j.*", "a.*"],
        body_name=["wrist", "pip", "pip_2", "pip_3", "thumb_fingertip", "tip", "tip_2", "tip_3"],
        controller=MultiConstraintDifferentialIKControllerCfg(
            command_type="position", use_relative_mode=False, ik_method="dls"
        ),
        scale=1,
        body_offset=MultiConstraintsDifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0, 0)
        ),
    )


@configclass
class RobotActionsCfg_IkAbsoluteDls:
    index_finger = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["j.*", "a.*"],
        body_name="wrist",
        controller=MultiConstraintDifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls"
        ),
        scale=1,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0, 0)
        ),
    )


"""
ROBOT OBSERVATIONS POLICY CFG
"""


@configclass
class RobotObservationsPolicyCfg:
    """Observations policy terms for the Scene."""

    robot_actions = ObsTerm(func=lab_mdp.last_action)
    robot_joint_pos = ObsTerm(func=lab_mdp.joint_pos_rel)
    robot_joint_vel = ObsTerm(func=lab_mdp.joint_vel_rel)
    # robot_end_effector_speed = ObsTerm(func=robot_mdp.end_effector_speed,
    #                                   params={"end_effector_speed_str": "end_effector_speed"})


"""
ROBOT COMMANDS CFG
"""


@configclass
class RobotCommandsCfg:
    """Command terms for the Scene."""

    pass


"""
ROBOT RANDOMIZATIONS CFG
"""


@configclass
class RobotRandomizationCfg:
    """Configuration for randomization."""

    reset_robot_joint = EventTerm(
        func=leap_xarm_mdp.reset_joints_by_offset,
        params={"asset_cfg": SceneEntityCfg("robot"),
                "position_range": [0.0, 1.5],
                "velocity_range": [-0.1, 0.1],
                "joint_ids": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]},
        mode="reset")


"""
ROBOT REWARDS CFG
"""


@configclass
class RobotRewardsCfg:
    """Reward terms for the MDP."""

    # action penalty
    action_rate = RewTerm(func=lab_mdp.action_rate_l2, weight=-0.01)

    joint_vel = RewTerm(
        func=lab_mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # reward_cross_finger_similarity = RewTerm(
    #     func=leap_xarm_mdp.reward_cross_finger_similarity,
    #     weight=0.1,
    #     params={"robot_cfg": SceneEntityCfg("robot")},
    # )

    # reward_intra_finger_similarity = RewTerm(
    #     func=leap_xarm_mdp.reward_intra_finger_similarity,
    #     weight=0.1,
    #     params={"robot_cfg": SceneEntityCfg("robot")},
    # )


"""
ROBOT TERMINATIONS CFG
"""


@configclass
class RobotTerminationsCfg:
    """Termination terms for the MDP."""

    pass
