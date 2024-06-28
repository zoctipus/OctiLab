# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
# from omni.isaac.lab.envs import RLTaskEnvCfg
from octilab.envs import OctiManagerBasedRLEnvCfg
# import mdp as tycho_mdp
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
import omni.isaac.lab.envs.mdp as orbit_mdp
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from octilab.managers.manager_term_cfg import DataTermCfg as DataTerm
from octilab.managers.manager_term_cfg import DataGroupCfg as DataGroup

from dataclasses import MISSING
##
# Pre-defined configs
##
from ... import mdp as task_mdp
from lab.cfgs.scenes.craneberryLavaChocoCake_scene import SceneObjectSceneCfg, SceneCommandsCfg, SceneEventCfg, SceneRewardsCfg, SceneTerminationsCfg
import lab.cfgs.robots.hebi.mdp as hebi_mdp
import lab.cfgs.robots.hebi.robot_dynamics as rd
from lab.cfgs.robots.hebi.robot_cfg import FRAME_EE, CAMERA_WRIST, CAMERA_BASE  # noqa: F401


@configclass
class ObjectSceneCfg():
    pass


@configclass
class IdealPDHebi_ObjectSceneCfg(SceneObjectSceneCfg, rd.RobotSceneCfg_IdealPD, ObjectSceneCfg):
    ee_frame = FRAME_EE


@configclass
class PwmMotorHebi_ObjectSceneCfg(SceneObjectSceneCfg, rd.RobotSceneCfg_HebiStrategy3Actuator, ObjectSceneCfg):
    ee_frame = FRAME_EE


@configclass
class ImplicitMotorHebi_ObjectSceneCfg(SceneObjectSceneCfg, rd.RobotSceneCfg_ImplicityActuator, ObjectSceneCfg):
    ee_frame = FRAME_EE
    # camera_wrist = CAMERA_WRIST
    # camera_base = CAMERA_BASE


@configclass
class DataCfg:
    # """Curriculum terms for the MDP."""
    @configclass
    class TaskData(DataGroup):
        data = DataTerm(func=task_mdp.update_data,
                        params={
                            "robot_name": "robot",
                            "canberry_name": "canberry",
                            "cake_name": "cake",
                            "caneberry_offset": [0, 0, 0],
                            "cake_offset": [0.0002, -0.0014, 0.0215],
                            "fixed_chop_frame_name": "frame_fixed_chop_tip",
                            "free_chop_frame_name": "frame_free_chop_tip"}
                        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    @configclass
    class HistoryData(DataGroup):
        history = DataTerm(func=task_mdp.update_history,
                           params={"robot_name": "robot"},
                           history_length=10)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    data: TaskData = TaskData()
    history: HistoryData = HistoryData()


@configclass
class ActionsCfg():
    '''Action will be defined at the moment class gets constructed'''
    pass


@configclass
class ObservationsCfg:
    ""
    @configclass
    class PolicyCfg(ObsGroup, rd.RobotObservationsPolicyCfg):
        """Observations for policy group."""
        canberry_pos_b = ObsTerm(func=hebi_mdp.position_in_robot_root_frame, params={"position_b": "canberry_position_b"})
        cake_pos_b = ObsTerm(func=hebi_mdp.position_in_robot_root_frame, params={"position_b": "cake_position_b"})
        # wrist_picture = ObsTerm(func=hebi_mdp.capture_image, params={"camera_key": "camera_wrist"})
        # base_picture = ObsTerm(func=hebi_mdp.capture_image, params={"camera_key": "camera_base"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg(SceneRewardsCfg, rd.RobotRewardsCfg):
    """Reward terms for the MDP."""

    reward_canberry_eeframe_distance_reward = RewTerm(
        func=task_mdp.reward_canberry_eeframe_distance_reward, 
        params={"canberry_eeframe_distance_key": "canberry_eeframe_distance"},
        weight=2,
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
        params={"canberry_eeframe_distance_key": "canberry_eeframe_distance",
                "chop_pose_key" : "chop_pose"},
        weight=1,
    )

    lift_reward = RewTerm(
        func=task_mdp.lift_reward,
        params={"canberry_height_key": "canberry_height",
                "canberry_grasp_mask_key": "canberry_grasp_mask"},
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
        params={
            "canberry_cake_distance_key" : "canberry_cake_distance",
            "chop_pose_key" : "chop_pose"},
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
class CommandsCfg(SceneCommandsCfg):
    """Command terms for the MDP."""
    pass


@configclass
class EventCfg(SceneEventCfg, rd.RobotRandomizationCfg):
    pass
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
class TerminationsCfg(SceneTerminationsCfg, rd.RobotTerminationsCfg):
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=orbit_mdp.time_out, time_out=True)
    success_state = DoneTerm(
        func=task_mdp.success_state,
        params={
            "canberry_cake_distance_key": "canberry_cake_distance",
            "canberry_eeframe_distance_key": "canberry_eeframe_distance",
            "chop_pose_key": "chop_pose",
        }
    )

    # berry_dropped = DoneTerm(
    #     func=task_mdp.canberry_dropped,
    #     params={
    #         "canberry_cake_distance_key":"canberry_cake_distance",
    #         "canberry_eeframe_distance_key":"canberry_eeframe_distance",
    #         "chop_pose_key":"chop_pose",
    #     }
    # )

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
        func=orbit_mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )


@configclass
class IdealPDHebi_JointPos_CraneberryLavaChocoCake_Env(OctiManagerBasedRLEnvCfg):
    # Scene settings
    scene: ObjectSceneCfg = IdealPDHebi_ObjectSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)
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
        self.sim.dt = 0.02 / self.decimation   # Agent: 20Hz, Motor: 500Hz
        self.sim.physx.min_position_iteration_count = 32
        self.sim.physx.min_velocity_iteration_count = 16
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2 ** 22
        self.sim.physx.gpu_max_rigid_patch_count = 2 ** 20


class PwmMotorHebi_JointPos_CraneberryLavaChocoCake_Env(IdealPDHebi_JointPos_CraneberryLavaChocoCake_Env):
    def __post_init__(self):
        super().__post_init__()
        self.scene: ObjectSceneCfg = PwmMotorHebi_ObjectSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)


class ImplicitMotorHebi_JointPos_CraneberryLavaChocoCake_Env(IdealPDHebi_JointPos_CraneberryLavaChocoCake_Env):
    def __post_init__(self):
        super().__post_init__()
        self.scene: ObjectSceneCfg = ImplicitMotorHebi_ObjectSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)
