# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from dataclasses import MISSING
from omni.isaac.lab.utils import configclass

from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm  # noqa: F401
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from octi.lab.envs import OctiManagerBasedRLEnvCfg
import octi.lab_tasks.tasks.manipulation.lift_objects.mdp as mdp
from ... import lift_cube
import octi.lab_assets.tycho.mdp as tycho_mdp
import octi.lab_assets.tycho as tycho
##
# Pre-defined configs
##


@configclass
class ObjectSceneCfg(lift_cube.ObjectSceneCfg):
    ee_frame = tycho.FRAME_EE
    robot = tycho.HEBI_IMPLICITY_ACTUATOR_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 0),
            rot=(0, 0, 0, 1),
            joint_pos=tycho.HEBI_DEFAULT_JOINTPOS
        ),)
    frame_fixed_chop_tip = tycho.FRAME_FIXED_CHOP_TIP
    frame_free_chop_tip = tycho.FRAME_FREE_CHOP_TIP
    # camera_wrist = CAMERA_WRIST
    # camera_base = CAMERA_BASE


@configclass
class DataCfg:
    # """Curriculum terms for the MDP."""
    pass


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    pass

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    reward_object_ee_distance = RewTerm(
        func=mdp.reward_object_ee_distance,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "fixed_chop_frame_cfg": SceneEntityCfg("frame_free_chop_tip"),
            "free_chop_frame_cfg": SceneEntityCfg("frame_free_chop_tip"),
        },
        weight=1.0,
    )
    
    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.1}, weight=15.0)

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.1, "command_name": "object_pose"},
        weight=30.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.1, "command_name": "object_pose"},
        weight=10.0,
    )


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="end_effector",
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-0.7, -0.5), pos_y=(-0.25, 0.25), pos_z=(0.1, 0.2), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )


@configclass
class EventCfg(lift_cube.EventCfg):
    reset_robot = EventTerm(
        func=mdp.reset_robot_to_default, params={"robot_cfg": SceneEntityCfg("robot")}, mode="reset"
    )


@configclass
class TerminationsCfg(lift_cube.TerminationsCfg):
    """Termination terms for the MDP."""
    robot_invalid_state = DoneTerm(func=mdp.invalid_state, params={"asset_cfg": SceneEntityCfg("robot")})

    robot_extremely_bad_posture = DoneTerm(
        func=tycho_mdp.terminate_extremely_bad_posture,
        params={"probability": 0.5, "robot_cfg": SceneEntityCfg("robot")},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


@configclass
class LiftCubeTycho(OctiManagerBasedRLEnvCfg):

    scene: ObjectSceneCfg = ObjectSceneCfg(num_envs=2048, env_spacing=2.5, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    datas: DataCfg = DataCfg()
    actions: ActionsCfg = MISSING
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 1
        self.episode_length_s = 5.0
        # simulation settings
        self.sim.dt = 0.01 / self.decimation
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024


@configclass
class LiftCubeTychoIkdelta(LiftCubeTycho):
    actions = tycho.IkdeltaAction()


@configclass
class LiftCubeTychoIkabsolute(LiftCubeTycho):
    actions = tycho.IkabsoluteAction()


@configclass
class LiftCubeTychoJointPosition(LiftCubeTycho):
    actions = tycho.JointPositionAction()


@configclass
class LiftCubeTychoJointEffort(LiftCubeTycho):
    actions = tycho.JointEffortAction()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = tycho.HEBI_EFFORT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
