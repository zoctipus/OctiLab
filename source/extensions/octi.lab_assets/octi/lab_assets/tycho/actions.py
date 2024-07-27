from omni.isaac.lab.envs.mdp.actions.actions_cfg import (
    JointPositionActionCfg,
    JointEffortActionCfg,
    DifferentialInverseKinematicsActionCfg,
    BinaryJointPositionActionCfg,
)
from omni.isaac.lab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from omni.isaac.lab.utils import configclass

JOINT_POSITION: JointPositionActionCfg = JointPositionActionCfg(
    asset_name="robot",
    joint_names=[
        "HEBI_base_X8_9",
        "HEBI_shoulder_X8_16",
        "HEBI_elbow_X8_9",
        "HEBI_wrist1_X5_1",
        "HEBI_wrist2_X5_1",
        "HEBI_wrist3_X5_1",
        "HEBI_chopstick_X5_1",
    ],
    scale=1,
)


JOINT_EFFORT: JointEffortActionCfg = JointEffortActionCfg(
    asset_name="robot",
    joint_names=[
        "HEBI_base_X8_9",
        "HEBI_shoulder_X8_16",
        "HEBI_elbow_X8_9",
        "HEBI_wrist1_X5_1",
        "HEBI_wrist2_X5_1",
        "HEBI_wrist3_X5_1",
        "HEBI_chopstick_X5_1",
    ],
    scale=0.1,
)


IKDELTA: DifferentialInverseKinematicsActionCfg = DifferentialInverseKinematicsActionCfg(
    asset_name="robot",
    joint_names=[
        "HEBI_base_X8_9",
        "HEBI_shoulder_X8_16",
        "HEBI_elbow_X8_9",
        "HEBI_wrist1_X5_1",
        "HEBI_wrist2_X5_1",
        "HEBI_wrist3_X5_1",
    ],
    body_name="static_chop_tip",
    controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
    scale=0.05,
    body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0, 0)),
)


IKABSOLUTE: DifferentialInverseKinematicsActionCfg = DifferentialInverseKinematicsActionCfg(
    asset_name="robot",
    joint_names=[
        "HEBI_base_X8_9",
        "HEBI_shoulder_X8_16",
        "HEBI_elbow_X8_9",
        "HEBI_wrist1_X5_1",
        "HEBI_wrist2_X5_1",
        "HEBI_wrist3_X5_1",
    ],
    body_name="static_chop_tip",  # Do not work if this is not end_effector
    controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
    scale=0.8,
    # while specifying position offset will work for most of case, but since rotation fluctuate as rotation
    # chop rotates this will also causes position fluctuation if position offset is not 0.
    # the most stable way will be having position offset be 0
    body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1, 0, 0, 0)),
)


BINARY_GRIPPER = BinaryJointPositionActionCfg(
    asset_name="robot",
    joint_names=["HEBI_chopstick_X5_1"],
    open_command_expr={"HEBI_chopstick_X5_1": -0.175},
    close_command_expr={"HEBI_chopstick_X5_1": -0.646},
)


@configclass
class IkdeltaAction:
    gripper_joint_pos = BINARY_GRIPPER
    body_joint_pos = IKDELTA


@configclass
class IkabsoluteAction:
    gripper_joint_pos = BINARY_GRIPPER
    body_joint_pos = IKABSOLUTE


@configclass
class JointPositionAction:
    jointpos = JOINT_POSITION


@configclass
class JointEffortAction:
    jointpos = JOINT_EFFORT
