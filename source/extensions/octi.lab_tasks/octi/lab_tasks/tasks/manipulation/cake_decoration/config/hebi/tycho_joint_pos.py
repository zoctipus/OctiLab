# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
import omni.isaac.lab.sim as sim_utils

from octi.lab.envs import OctiManagerBasedRLEnvCfg
from ... import cake_decoration_env as CakeDecorationEnv
from omni.isaac.lab.managers import SceneEntityCfg

from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from octi.lab.managers.manager_term_cfg import DataTermCfg as DataTerm
from octi.lab.managers.manager_term_cfg import DataGroupCfg as DataGroup

from dataclasses import MISSING

##
# Pre-defined configs
##
from ... import mdp as task_mdp

# from octi.lab_tasks.cfgs.robots.hebi.robot_cfg import FRAME_EE, CAMERA_WRIST, CAMERA_BASE  # noqa: F401
from octi.lab_assets.tycho import FRAME_EE, CAMERA_WRIST, CAMERA_BASE, FRAME_FIXED_CHOP_TIP, FRAME_FREE_CHOP_TIP  # noqa: F401
from octi.lab_assets.tycho import HEBI_IMPLICITY_ACTUATOR_CFG, HEBI_EFFORT_CFG
from octi.lab_assets.tycho import IKDELTA, IKABSOLUTE, JOINT_POSITION, BINARY_GRIPPER, JOINT_EFFORT

RADIUS = 0.02

plate_position = [-0.5, -0.3, 0.04]
plate_scale = 1.5

cake_position = [-0.2, -0.48, 0.00362]
cake_scale = 5

canberry_position = [plate_position[0] + 0.06, plate_position[1] + 0.02, plate_position[2] + 0.06]
canberry_scale = 5

canberryTree_position = [plate_position[0] + 0.03, plate_position[1] + 0.07, plate_position[2]]
canberryTree_scale = 5

p_glassware_short_position = [cake_position[0] + 0.2, cake_position[1] + 0.1, cake_position[2] + 0.1]
p_glassware_short_scale = 1

spoon_position = [0.016113101099947867, -0.47999998927116405, 0.019141643538828867]
spoon_scale = 3


@configclass
class ObjectSceneCfg(CakeDecorationEnv.ObjectSceneCfg):
    ee_frame = FRAME_EE
    robot = HEBI_IMPLICITY_ACTUATOR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    frame_fixed_chop_tip = FRAME_FIXED_CHOP_TIP
    frame_free_chop_tip = FRAME_FREE_CHOP_TIP


@configclass
class DataCfg:
    # """Curriculum terms for the MDP."""
    @configclass
    class TaskData(DataGroup):
        data = DataTerm(
            func=task_mdp.update_data,
            params={"robot_name": "robot",
                    "canberry_name": "canberry",
                    "cake_name": "cake",
                    "caneberry_offset": [0, 0, 0],
                    "cake_offset": [0.0002, -0.0014, 0.0215],
                    "fixed_chop_frame_name": "frame_fixed_chop_tip",
                    "free_chop_frame_name": "frame_free_chop_tip"},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    @configclass
    class HistoryData(DataGroup):
        history = DataTerm(func=task_mdp.update_history, params={"robot_name": "robot"}, history_length=10)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    data: TaskData = TaskData()
    history: HistoryData = HistoryData()


@configclass
class ActionsCfg:
    pass


@configclass
class ObservationsCfg():
    """"""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        robot_actions = ObsTerm(func=task_mdp.last_action)
        robot_joint_pos = ObsTerm(func=task_mdp.joint_pos_rel)
        robot_joint_vel = ObsTerm(func=task_mdp.joint_vel_rel)
        robot_eepose = ObsTerm(func=task_mdp.end_effector_pose_in_robot_root_frame)
        canberry_pos_b = ObsTerm(
            func=task_mdp.position_in_robot_root_frame, params={"position_b": "canberry_position_b"}
        )
        cake_pos_b = ObsTerm(func=task_mdp.position_in_robot_root_frame, params={"position_b": "cake_position_b"})
        # wrist_picture = ObsTerm(func=hebi_mdp.capture_image, params={"camera_key": "camera_wrist"})
        # base_picture = ObsTerm(func=hebi_mdp.capture_image, params={"camera_key": "camera_base"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg(CakeDecorationEnv.RewardsCfg):
    """Reward terms for the MDP."""

    action_rate = RewTerm(func=task_mdp.action_rate_l2, weight=-0.01)

    joint_vel = RewTerm(
        func=task_mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    velocity_punishment = RewTerm(func=task_mdp.joint_vel_limits, weight=-1, params={"soft_ratio": 1})

    punish_hand_tilted = RewTerm(func=task_mdp.punish_hand_tilted, weight=1 )

    punish_touching_ground = RewTerm(func=task_mdp.punish_touching_ground, weight=0.3)

    punish_bad_elbow_posture = RewTerm(func=task_mdp.punish_bad_elbow_shoulder_posture, weight=1)

    reward_canberry_eeframe_distance_reward = RewTerm(
        func=task_mdp.reward_canberry_eeframe_distance_reward,
        params={"canberry_eeframe_distance_key": "canberry_eeframe_distance"},
        weight=2
    )

    try_to_close_reward = RewTerm(
        func=task_mdp.try_to_close_reward,
        params={"canberry_eeframe_distance_key": "canberry_eeframe_distance",
                "chop_pose_key": "chop_pose",
                "chop_tips_canberry_cos_angle_key": "chop_tips_canberry_cos_angle"},
        weight=2,
    )

    miss_penalty = RewTerm(
        func=task_mdp.miss_penalty,
        params={"canberry_eeframe_distance_key": "canberry_eeframe_distance", "chop_pose_key": "chop_pose"},
        weight=1,
    )

    lift_reward = RewTerm(
        func=task_mdp.lift_reward,
        params={"canberry_height_key": "canberry_height", "canberry_grasp_mask_key": "canberry_grasp_mask"},
        weight=5,
    )

    chop_action_rate_penalty = RewTerm(
        func=task_mdp.chop_action_rate_l2,
        params={},
        weight=1,
    )

    canberry_cake_distance_reward = RewTerm(
        func=task_mdp.canberry_cake_distance_reward,
        params={"canberry_cake_distance_key": "canberry_cake_distance",
                "canberry_height_key": "canberry_height",
                "canberry_grasp_mask_key": "canberry_grasp_mask"},
        weight=5,
    )

    try_to_drop_reward = RewTerm(
        func=task_mdp.try_to_drop_reward,
        params={"canberry_cake_distance_key": "canberry_cake_distance", "chop_pose_key": "chop_pose"},
        weight=30,
    )

    # success_reward = RewTerm(
    #     func=task_mdp.success_reward, params=
    #         {
    #             "chop_pose_key" : "chop_pose",
    #             "canberry_cake_distance_key" : "canberry_cake_distance"
    #         },
    #     weight=1000000,
    # )


@configclass
class CommandsCfg(CakeDecorationEnv.CommandsCfg):
    """Command terms for the MDP."""

    pass


@configclass
class EventCfg(CakeDecorationEnv.EventCfg):

    # Reset
    reset_robot = EventTerm(
        func=task_mdp.reset_robot_to_default, params={"robot_cfg": SceneEntityCfg("robot")}, mode="reset"
    )
    # record_state_configuration = EventTerm(
    #     func=task_mdp.record_state_configuration,
    #     mode="interval",
    #     interval_range_s=(0.5, 1),
    #     params={},
    # )

    # reset_from_demostration = EventTerm(
    #     func=task_mdp.reset_from_demostration,
    #     mode="reset",
    #     params={},
    # )


@configclass
class TerminationsCfg(CakeDecorationEnv.TerminationsCfg):
    """Termination terms for the MDP."""
    success_state = DoneTerm(
        func=task_mdp.success_state,
        params={
            "canberry_cake_distance_key": "canberry_cake_distance",
            "canberry_eeframe_distance_key": "canberry_eeframe_distance",
            "chop_pose_key": "chop_pose",
        },
    )

    robot_invalid_state = DoneTerm(func=task_mdp.invalid_state, params={"asset_cfg": SceneEntityCfg("robot")})

    robot_extremely_bad_posture = DoneTerm(
        func=task_mdp.terminate_extremely_bad_posture,
        params={"probability": 0.5, "robot_cfg": SceneEntityCfg("robot")},
    )

    # non_moving_abnorm = DoneTerm(
    #     func=task_mdp.non_moving_abnormalty,
    #     params={
    #         "end_effector_speed_str": "end_effector_speed"
    #     }
    # )

    # chop_missing = DoneTerm(
    #     func=task_mdp.chop_missing,
    #     params={
    #         "canberry_eeframe_distance_key":"canberry_eeframe_distance",
    #         "chop_pose_key":"chop_pose",
    #     }
    # )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    change_drop_weight = CurrTerm(
        func=task_mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )


@configclass
class CakeDecorationTychoEnv(OctiManagerBasedRLEnvCfg):
    # Scene settings
    scene: ObjectSceneCfg = ObjectSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = MISSING  # type: ignore
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    datas: DataCfg = DataCfg()

    def __post_init__(self):
        self.decimation = 1
        self.episode_length_s = 4
        # simulation settings
        self.sim.dt = 0.02 / self.decimation  # Agent: 20Hz, Motor: 500Hz
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**22
        self.sim.physx.gpu_max_rigid_patch_count = 2**20


@configclass
class IkdeltaAction(ActionsCfg):
    gripper_joint_pos = BINARY_GRIPPER
    body_joint_pos = IKDELTA


@configclass
class IkabsoluteAction(ActionsCfg):
    gripper_joint_pos = BINARY_GRIPPER
    body_joint_pos = IKABSOLUTE


@configclass
class JointPositionAction(ActionsCfg):
    jointpos = JOINT_POSITION


@configclass
class JointEffortAction(ActionsCfg):
    jointpos = JOINT_EFFORT


@configclass
class CakeDecorationTychoIkdelta(CakeDecorationTychoEnv):
    actions = IkdeltaAction()


@configclass
class CakeDecorationTychoIkabsolute(CakeDecorationTychoEnv):
    actions = IkabsoluteAction()


@configclass
class CakeDecorationTychoJointPosition(CakeDecorationTychoEnv):
    actions = JointPositionAction()


@configclass
class CakeDecorationTychoJointEffort(CakeDecorationTychoEnv):
    actions = JointEffortAction()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = HEBI_EFFORT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
