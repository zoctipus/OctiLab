# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from omni.isaac.lab.utils import configclass
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import AssetBaseCfg, ArticulationCfg
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.envs.mdp import time_out
from dataclasses import MISSING
from . import mdp

class SceneCfg(InteractiveSceneCfg):

    robot: ArticulationCfg = MISSING  # type: ignore
    
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, 0)),
        spawn=sim_utils.GroundPlaneCfg(),
    )
    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=1500.0, color=(0.75, 0.75, 0.75))
    )


@configclass
class ObservationsCfg:
    """"""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    pass


@configclass
class CommandsCfg:
    """Command terms for the MDP."""
    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="REPLACE_ME",
        resampling_time_range=(10, 10),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.1, 0.45),
            pos_y=(-0.4, 0.4),
            pos_z=(0.02, 0.3),
            roll=(0, 0.0),
            pitch=(0, 1),
            yaw=(0.0, 1),
        ),
    )
    

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    end_effector_position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="REPLACE_ME"),
            "std": 0.5,
            "command_name": "ee_pose",
        },
    )

    end_effector_position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="REPLACE_ME"),
            "std": 0.1,
            "command_name": "ee_pose",
        },
    )

    end_effector_orientation_tracking = RewTerm(
        func=mdp.orientation_command_error_tanh,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="REPLACE_ME"),
            "std": 0.5,
            "command_name": "ee_pose",
        },
    )

@configclass
class DataCfg:
    pass


@configclass
class EventCfg:
    reset_robot_joint = EventTerm(
        func=mdp.reset_joints_by_offset,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": [0.0, 1.5],
            "velocity_range": [-0.1, 0.1],
        },
        mode="reset",
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=time_out, time_out=True)


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    pass


@configclass
class TrackGoalEnv(ManagerBasedRLEnvCfg):

    # Scene settings
    scene: SceneCfg = SceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)
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
        self.episode_length_s = 50
        # simulation settings
        self.sim.dt = 0.02 / self.decimation
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.gpu_max_rigid_patch_count = 5 * 2 ** 16