from __future__ import annotations

from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
import omni.isaac.lab.envs.mdp as orbit_mdp

'''
ROBOT ACTION CFGS
'''
from omni.isaac.lab.envs.mdp.actions.actions_cfg import (JointPositionActionCfg,
                                                         JointEffortActionCfg,
                                                         DifferentialInverseKinematicsActionCfg)
from octilab.envs.mdp.actions.actions_cfg import MultiConstraintsDifferentialInverseKinematicsActionCfg
from omni.isaac.lab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from octilab.controllers.differential_ik_cfg import MultiConstraintDifferentialIKControllerCfg
from . import (  # noqa: F401
    IMPLICIT_LEAP
)


@configclass
class RobotSceneCfg_ImplicityActuator:
    '''
    HEBI Implicity Actuator is a simple low PD that groups "X8_9", "X8_16", "x5" typed
    motor into identical PD values.
    '''
    robot = IMPLICIT_LEAP.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class RobotActionsCfg_JointPosition:
    """Action specifications for the MDP."""
    joint_position: JointPositionActionCfg = JointPositionActionCfg(
        asset_name="robot", joint_names=["j.*"], scale=1)


@configclass
class RobotActionsCfg_JointEffort:
    """Action specifications for the MDP."""
    joint_efforts: JointEffortActionCfg = JointEffortActionCfg(
        asset_name="robot", joint_names=["j.*"], scale=0.1
    )


@configclass
class RobotActionsCfg_IkDeltaDls:
    index_finger = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["j[0-3]", "w[0-5]"],
        body_name="fingertip",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=0.05,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0, 0)),
    )


@configclass
class RobotActionsCfg_IkAbsoluteDls:
    index_finger = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["w[0-5]"],
        body_name="palm_lower",
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls"),
        scale=1,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0, 0)),
    )


@configclass
class RobotActionsCfg_MCIkAbsoluteDls:
    index_finger = MultiConstraintsDifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["w.*", "j.*"],
        # body_name=["wrist", "fingertip", "thumb_fingertip", "fingertip_2", "fingertip_3"],
        body_name=["wrist", "thumb_fingertip", "tip", "tip_2", "tip_3"],
        controller=MultiConstraintDifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls"),
        scale=1,
        body_offset=MultiConstraintsDifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0, 0)),
    )


'''
ROBOT OBSERVATIONS POLICY CFG
'''


@configclass
class RobotObservationsPolicyCfg:
    """Observations policy terms for the Scene."""
    robot_actions = ObsTerm(func=orbit_mdp.last_action)
    robot_joint_pos = ObsTerm(func=orbit_mdp.joint_pos_rel)
    robot_joint_vel = ObsTerm(func=orbit_mdp.joint_vel_rel)
    # robot_end_effector_speed = ObsTerm(func=robot_mdp.end_effector_speed,
    #                                   params={"end_effector_speed_str": "end_effector_speed"})


'''
ROBOT COMMANDS CFG
'''


@configclass
class RobotCommandsCfg:
    """Command terms for the Scene."""
    pass


'''
ROBOT RANDOMIZATIONS CFG
'''


@configclass
class RobotRandomizationCfg:
    """Configuration for randomization."""
    reset_robot = EventTerm(
        func=orbit_mdp.reset_scene_to_default,
        params={},
        mode="reset")


'''
ROBOT REWARDS CFG
'''


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


'''
ROBOT TERMINATIONS CFG
'''


@configclass
class RobotTerminationsCfg:
    """Termination terms for the MDP."""
    pass
