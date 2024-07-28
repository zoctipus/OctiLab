# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import SceneEntityCfg

# import mdp as tycho_mdp
import omni.isaac.lab.envs.mdp as lab_mdp
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from dataclasses import MISSING

##
# Pre-defined configs
##
import omni.isaac.lab_tasks.manager_based.manipulation.reach.mdp as mdp
import octi.lab_tasks.tasks.manipulation.track_goal.mdp as task_mdp
# from octi.lab_tasks.cfgs.robots.hebi.robot_dynamics import RobotRewardsCfg
from ... import track_goal_env
# import octi.lab_tasks.cfgs.robots.hebi.mdp as hebimdp
import octi.lab_assets.tycho as tycho

episode_length = 50.0


class ObjectSceneCfg(track_goal_env.ObjectSceneCfg):
    ee_frame = tycho.FRAME_EE
    robot = tycho.HEBI_IMPLICITY_ACTUATOR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    frame_fixed_chop_tip = tycho.FRAME_FIXED_CHOP_TIP
    frame_free_chop_tip = tycho.FRAME_FREE_CHOP_TIP


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    pass


@configclass
class ObservationsCfg:
    """"""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=lab_mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=lab_mdp.joint_vel_rel)
        target_object_position = ObsTerm(func=lab_mdp.generated_commands, params={"command_name": "ee_pose"})
        actions = ObsTerm(func=lab_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # action penalty
    action_rate = RewTerm(func=task_mdp.action_rate_l2, weight=-0.01)

    joint_vel = RewTerm(
        func=task_mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    velocity_punishment = RewTerm(func=task_mdp.joint_vel_limits, weight=-1, params={"soft_ratio": 1})

    punish_hand_tilted = RewTerm(
        func=task_mdp.punish_hand_tilted,
        weight=1,
    )

    punish_touching_ground = RewTerm(
        func=task_mdp.punish_touching_ground,
        weight=0.3,
    )

    punish_bad_elbow_posture = RewTerm(
        func=task_mdp.punish_bad_elbow_shoulder_posture,
        weight=1,
    )

    end_effector_position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="static_chop_tip"),
            "std": 0.5,
            "command_name": "ee_pose",
        },
    )

    end_effector_position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=4,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="static_chop_tip"),
            "std": 0.1,
            "command_name": "ee_pose",
        },
    )

    end_effector_orientation_tracking = RewTerm(
        func=hebimdp.orientation_command_error_tanh,
        weight=2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="static_chop_tip"),
            "std": 0.5,
            "command_name": "ee_pose",
        },
    )


@configclass
class DataCfg:
    # """Curriculum terms for the MDP."""
    pass


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    ee_pose = lab_mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="static_chop_tip",
        resampling_time_range=(episode_length / 5, episode_length / 5),
        debug_vis=False,
        ranges=lab_mdp.UniformPoseCommandCfg.Ranges(
            # CAUTION: to use the hebi's ik solver the roll needs to be (0.0, 0.0) for upright vector to be correct
            # however, for the isaac lab ik solver, the roll needs to be (1.57075, 1.57075)
            pos_x=(-0.45, -0.15),
            pos_y=(-0.5, -0.15),
            pos_z=(0.02, 0.3),
            roll=(1.57075, 1.57075),
            pitch=(3.14, 3.14),
            yaw=(0.0, 0.5),
        ),
    )


@configclass
class EventCfg:
    reset_robot = EventTerm(
        func=task_mdp.reset_robot_to_default, params={"robot_cfg": SceneEntityCfg("robot")}, mode="reset"
    )


@configclass
class TerminationsCfg(track_goal_env.TerminationsCfg):
    """Termination terms for the MDP."""
    robot_invalid_state = DoneTerm(func=task_mdp.invalid_state, params={"asset_cfg": SceneEntityCfg("robot")})

    robot_extremely_bad_posture = DoneTerm(
        func=task_mdp.terminate_extremely_bad_posture,
        params={"probability": 0.5, "robot_cfg": SceneEntityCfg("robot")},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


@configclass
class GoalTrackingTychoEnv(ManagerBasedRLEnvCfg):

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
        self.decimation = 2
        self.episode_length_s = episode_length
        # simulation settings
        self.sim.dt = 0.02 / self.decimation  # Env_Hz = Agent_Hz / decimation**2
        self.sim.physx.gpu_max_rigid_patch_count = 5 * 2**17


@configclass
class GoalTrackingTychoIkdelta(GoalTrackingTychoEnv):
    actions = tycho.IkdeltaAction


@configclass
class GoalTrackingTychoIkabsolute(GoalTrackingTychoEnv):
    actions = tycho.IkabsoluteAction


@configclass
class GoalTrackingTychoJointPosition(GoalTrackingTychoEnv):
    actions = tycho.JointPositionAction


@configclass
class GoalTrackingTychoJointEffort(GoalTrackingTychoEnv):
    actions = tycho.JointEffortAction

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = tycho.HEBI_EFFORT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
