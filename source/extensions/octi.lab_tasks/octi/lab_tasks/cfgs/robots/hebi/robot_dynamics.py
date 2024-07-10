from __future__ import annotations

from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
import omni.isaac.lab.envs.mdp as orbit_mdp
import octi.lab_tasks.cfgs.robots.hebi.mdp as robot_mdp
import octi.lab.envs.mdp as tycho_general_mdp

"""
ROBOT ACTION CFGS
"""
from omni.isaac.lab.envs.mdp.actions.actions_cfg import (
    JointPositionActionCfg,
    JointEffortActionCfg,
    DifferentialInverseKinematicsActionCfg,
    BinaryJointPositionActionCfg,
)

from omni.isaac.lab.controllers.differential_ik_cfg import DifferentialIKControllerCfg

from . import (  # noqa: F401
    HEBI_IMPLICITY_ACTUATOR_CFG,
    HEBI_STRATEGY3_CFG,
    HEBI_STRATEGY4_CFG,
    HEBI_IDEAL_PD_CFG,
    HEBI_EFFORT_CFG,
    HEBI_DCMOTOR_CFG,
    DCMOTOR_CFG,
    FRAME_FIXED_CHOP_TIP,
    FRAME_FIXED_CHOP_END,
    FRAME_FREE_CHOP_TIP,
    CAMERA_WRIST,
)


@configclass
class RobotSceneCfg_HebiEffort:
    robot = HEBI_EFFORT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    frame_fixed_chop_tip = FRAME_FIXED_CHOP_TIP
    frame_free_chop_tip = FRAME_FREE_CHOP_TIP


@configclass
class RobotSceneCfg_HebiDCMotor:
    robot = HEBI_DCMOTOR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    frame_fixed_chop_tip = FRAME_FIXED_CHOP_TIP
    frame_free_chop_tip = FRAME_FREE_CHOP_TIP


@configclass
class RobotSceneCfg_DCMotor:
    robot = DCMOTOR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    frame_fixed_chop_tip = FRAME_FIXED_CHOP_TIP
    frame_free_chop_tip = FRAME_FREE_CHOP_TIP


@configclass
class RobotSceneCfg_HebiStrategy3Actuator:
    """
    **Hebi Pwm Motor requires target joint position** and thus will only work with
    Actions that output desired JointPos, such us IK, JointPositionAction, etc
    and will not work with JointEffortAction or JointVelocityAction.

    Hebi Pwm Motor has no stiffness or damping, such values are defined in xml supplied
    by the PwmMotor configuration.
    """

    robot = HEBI_STRATEGY3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    frame_fixed_chop_tip = FRAME_FIXED_CHOP_TIP
    frame_free_chop_tip = FRAME_FREE_CHOP_TIP


@configclass
class RobotSceneCfg_HebiStrategy4Actuator:
    """
    **Hebi Pwm Motor requires target joint position** and thus will only work with
    Actions that output desired JointPos, such us IK, JointPositionAction, etc
    and will not work with JointEffortAction or JointVelocityAction.

    Hebi Pwm Motor has no stiffness or damping, such values are defined in xml supplied
    by the PwmMotor configuration.
    """

    robot = HEBI_STRATEGY4_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    frame_fixed_chop_tip = FRAME_FIXED_CHOP_TIP
    frame_free_chop_tip = FRAME_FREE_CHOP_TIP


@configclass
class RobotSceneCfg_ImplicityActuator:
    """
    HEBI Implicity Actuator is a simple low PD that groups "X8_9", "X8_16", "x5" typed
    motor into identical PD values.
    """

    robot = HEBI_IMPLICITY_ACTUATOR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    frame_fixed_chop_tip = FRAME_FIXED_CHOP_TIP
    frame_free_chop_tip = FRAME_FREE_CHOP_TIP


@configclass
class RobotSceneCfg_IdealPD:
    """
    HEBI HighPD Actuator is a simple high PD that groups "X8_9", "X8_16", "x5" typed
    motor into identical PD values.
    """

    robot = HEBI_IDEAL_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    frame_fixed_chop_tip = FRAME_FIXED_CHOP_TIP
    frame_free_chop_tip = FRAME_FREE_CHOP_TIP


@configclass
class RobotActionsCfg_HebiJointPosition:
    """Action specifications for the MDP."""

    joint_position: JointPositionActionCfg = JointPositionActionCfg(
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


@configclass
class RobotActionsCfg_HebiJointEffort:
    """Action specifications for the MDP."""

    joint_efforts: JointEffortActionCfg = JointEffortActionCfg(
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


@configclass
class RobotActionsCfg_HebiIkDeltaDls:
    body_joint_pos = DifferentialInverseKinematicsActionCfg(
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
    finger_joint_pos = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["HEBI_chopstick_X5_1"],
        open_command_expr={"HEBI_chopstick_X5_1": -0.175},
        close_command_expr={"HEBI_chopstick_X5_1": -0.646},
    )


@configclass
class RobotActionsCfg_HebiIkAbsoluteDls:
    body_joint_pos = DifferentialInverseKinematicsActionCfg(
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
        scale=1,
        # while specifying position offset will work for most of case, but since rotation fluctuate as rotation
        # chop rotates this will also causes position fluctuation if position offset is not 0.
        # the most stable way will be having position offset be 0
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1, 0, 0, 0)),
    )
    finger_joint_pos = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["HEBI_chopstick_X5_1"],
        open_command_expr={"HEBI_chopstick_X5_1": -0.2},
        close_command_expr={"HEBI_chopstick_X5_1": -0.6},
    )


@configclass
class RobotActionsCfg_OriginHebiIkAbsoluteDls:
    body_joint_pos = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=[
            "HEBI_base_X8_9",
            "HEBI_shoulder_X8_16",
            "HEBI_elbow_X8_9",
            "HEBI_wrist1_X5_1",
            "HEBI_wrist2_X5_1",
            "HEBI_wrist3_X5_1",
        ],
        body_name="HEBI_static_mount_chopstick_tip_link",  # Do not work if this is not end_effector
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        scale=1,
        # while specifying position offset will work for most of case, but since rotation fluctuate as rotation
        # chop rotates this will also causes position fluctuation if position offset is not 0.
        # the most stable way will be having position offset be 0
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0, 0)),
    )
    finger_joint_pos = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["HEBI_chopstick_actuator_X5_1"],
        open_command_expr={"HEBI_chopstick_actuator_X5_1": -0.3},
        close_command_expr={"HEBI_chopstick_actuator_X5_1": -0.5},
    )


"""
ROBOT OBSERVATIONS POLICY CFG
"""


@configclass
class RobotObservationsPolicyCfg:
    """Observations policy terms for the Scene."""

    robot_actions = ObsTerm(func=orbit_mdp.last_action)
    robot_joint_pos = ObsTerm(func=orbit_mdp.joint_pos_rel)
    robot_joint_vel = ObsTerm(func=orbit_mdp.joint_vel_rel)
    robot_eepose = ObsTerm(func=robot_mdp.end_effector_pose_in_robot_root_frame)
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

    reset_robot = EventTerm(
        func=tycho_general_mdp.reset_robot_to_default, params={"robot_cfg": SceneEntityCfg("robot")}, mode="reset"
    )


"""
ROBOT REWARDS CFG
"""


@configclass
class RobotRewardsCfg:
    """Reward terms for the MDP."""

    # action penalty
    action_rate = RewTerm(func=orbit_mdp.action_rate_l2, weight=-0.01)

    joint_vel = RewTerm(
        func=orbit_mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    velocity_punishment = RewTerm(func=orbit_mdp.joint_vel_limits, weight=-1, params={"soft_ratio": 1})

    punish_hand_tilted = RewTerm(
        func=robot_mdp.punish_hand_tilted,
        weight=1,
    )

    punish_touching_ground = RewTerm(
        func=robot_mdp.punish_touching_ground,
        weight=0.3,
    )

    punish_bad_elbow_posture = RewTerm(
        func=robot_mdp.punish_bad_elbow_shoulder_posture,
        weight=1,
    )


"""
ROBOT TERMINATIONS CFG
"""


@configclass
class RobotTerminationsCfg:
    """Termination terms for the MDP."""

    robot_invalid_state = DoneTerm(func=tycho_general_mdp.invalid_state, params={"asset_cfg": SceneEntityCfg("robot")})

    robot_extremely_bad_posture = DoneTerm(
        func=robot_mdp.terminate_extremely_bad_posture,
        params={"probability": 0.5, "robot_cfg": SceneEntityCfg("robot")},
    )
