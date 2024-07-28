from __future__ import annotations
from omni.isaac.lab.envs.mdp.actions.actions_cfg import (JointPositionActionCfg, JointEffortActionCfg)
from omni.isaac.lab.utils import configclass
from octi.lab.envs.mdp.actions.actions_cfg import MultiConstraintsDifferentialInverseKinematicsActionCfg
from octi.lab.controllers.differential_ik_cfg import MultiConstraintDifferentialIKControllerCfg
"""
LEAP ACTIONS
"""

LEAP_JOINT_POSITION: JointPositionActionCfg = JointPositionActionCfg(asset_name="robot", joint_names=["w.*", "j.*"], scale=1)


LEAP_JOINT_EFFORT: JointEffortActionCfg = JointEffortActionCfg(asset_name="robot", joint_names=["w.*", "j.*"], scale=0.1)


LEAP_MC_IKABSOLUTE = MultiConstraintsDifferentialInverseKinematicsActionCfg(
    asset_name="robot",
    joint_names=["w.*", "j.*"],
    body_name=["wrist", "pip", "pip_2", "pip_3", "thumb_fingertip", "tip", "tip_2", "tip_3"],
    controller=MultiConstraintDifferentialIKControllerCfg(
        command_type="position", use_relative_mode=False, ik_method="dls"
    ),
    scale=1,
    body_offset=MultiConstraintsDifferentialInverseKinematicsActionCfg.OffsetCfg(
        pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0, 0)
    ),
)

LEAP_MC_IKDELTA = MultiConstraintsDifferentialInverseKinematicsActionCfg(
    asset_name="robot",
    joint_names=["w.*", "j.*"],
    body_name=["wrist", "pip", "pip_2", "pip_3", "thumb_fingertip", "tip", "tip_2", "tip_3"],
    controller=MultiConstraintDifferentialIKControllerCfg(
        command_type="position", use_relative_mode=True, ik_method="dls"
    ),
    scale=0.1,
    body_offset=MultiConstraintsDifferentialInverseKinematicsActionCfg.OffsetCfg(
        pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0, 0)
    ),
)


@configclass
class LeapMcIkAbsoluteAction:
    jointpos = LEAP_MC_IKABSOLUTE


@configclass
class LeapMcIkDeltaAction:
    jointpos = LEAP_MC_IKDELTA


@configclass
class LeapJointPositionAction:
    jointpos = LEAP_JOINT_POSITION


@configclass
class LeapJointEffortAction:
    jointpos = LEAP_JOINT_EFFORT


"""
LEAP XARM ACTIONS
"""
LEAPXARM_JOINT_POSITION: JointPositionActionCfg = JointPositionActionCfg(
    asset_name="robot", joint_names=["j.*", "a.*"], scale=1
)


LEAPXARM_JOINT_EFFORT: JointEffortActionCfg = JointEffortActionCfg(
    asset_name="robot", joint_names=["j.*", "a.*"], scale=0.1
)


LEAPXARM_MC_IKABSOLUTE = MultiConstraintsDifferentialInverseKinematicsActionCfg(
    asset_name="robot",
    joint_names=["j.*", "a.*"],
    body_name=["wrist", "pip", "pip_2", "pip_3", "tip", "thumb_tip", "tip_2", "tip_3"],
    controller=MultiConstraintDifferentialIKControllerCfg(
        command_type="position", use_relative_mode=False, ik_method="dls"
    ),
    scale=1,
    body_offset=MultiConstraintsDifferentialInverseKinematicsActionCfg.OffsetCfg(
        pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0, 0)
    ),
)


LEAPXARM_MC_IKDELTA = MultiConstraintsDifferentialInverseKinematicsActionCfg(
    asset_name="robot",
    joint_names=["j.*", "a.*"],
    body_name=["wrist", "pip", "pip_2", "pip_3", "tip", "thumb_tip", "tip_2", "tip_3"],
    controller=MultiConstraintDifferentialIKControllerCfg(
        command_type="position", use_relative_mode=True, ik_method="dls"
    ),
    scale=1,
    body_offset=MultiConstraintsDifferentialInverseKinematicsActionCfg.OffsetCfg(
        pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0, 0)
    ),
)


@configclass
class LeapXarmMcIkAbsoluteAction:
    jointpos = LEAPXARM_MC_IKABSOLUTE


@configclass
class LeapXarmMcIkDeltaAction:
    jointpos = LEAPXARM_MC_IKDELTA


@configclass
class LeapXarmJointPositionAction:
    jointpos = LEAPXARM_JOINT_POSITION


@configclass
class LeapXarmJointEffortAction:
    jointpos = LEAPXARM_JOINT_EFFORT
