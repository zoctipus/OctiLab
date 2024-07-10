from __future__ import annotations
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.managers import RandomizationTermCfg as RandTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
import omni.isaac.lab.envs.mdp as orbit_mdp
import octi.lab.envs.mdp as tycho_general_mdp

from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip

marker_cfg = FRAME_MARKER_CFG.copy()
marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
marker_cfg.prim_path = "/Visuals/RopeFrameTransformer"


@configclass
class SceneObjectSceneCfg(InteractiveSceneCfg):
    """Configuration for a testing rope scene"""

    object: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path="assets/rope_test.usd",
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=1, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(-0.4, 0.0, 1.05)),
        actuators={"body": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0)},
    )

    object_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Object/capsule_0",
        debug_vis=True,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Object/Object",
                name="object_frame",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.0),
                ),
            ),
        ],
    )

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=1500.0, color=(0.75, 0.75, 0.75))
    )


##
# MDP setting
##


@configclass
class SceneCommandsCfg:
    """Command terms for the Scene."""

    pass


@configclass
class SceneRandomizationCfg:
    """Configuration for randomization."""

    reset_all = RandTerm(func=orbit_mdp.reset_scene_to_default, mode="reset")

    randomize_rope_positions = RandTerm(
        func=orbit_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.2, 0.2), "y": (-0.5, -0.2), "z": (0, 0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )

    # randomize_rigid_body_materials = RandTerm(
    #     func=orbit_mdp.randomize_rigid_body_material,
    #     mode='reset',
    #     params={
    #         "static_friction_range": (0.4, 1),
    #         "dynamic_friction_range": (0.4, 1),
    #         "restitution_range": (0, 0.1),
    #         "num_buckets": 2,
    #         "asset_cfg": SceneEntityCfg("object", body_names="Object"),
    #     }
    # )

    # add_rigid_body_mass = RandTerm(
    #     func=orbit_mdp.add_body_mass,
    #     mode="reset",
    #     params={
    #         "mass_range": (0.001, 0.01),
    #         "asset_cfg": SceneEntityCfg("object", body_names="Object"),
    #     }
    # )

    # randomize_joint_physical_property = RandTerm(
    #     func=mdp.randomize_joint_physical_property,
    #     mode="reset",
    #     params={
    #         "stiffness_range" : {"X8_9":(86, 88), "X8_16":(79, 81), "x5":(1900, 2100)},
    #         "damping_range" : {"X8_9":(39, 41), "X8_16":(39, 41), "x5":(2,4)},
    #         "armature_range" : {"X8_9":(0.0009, 0.0011), "X8_16":(0.0009, 0.0011), "x5":(0.0009, 0.0011)},
    #         "friction_range" : {"X8_9":(0.19, 0.21), "X8_16":(0.19, 0.21), "x5":(0.19, 0.21)},
    #         "asset_cfg": SceneEntityCfg("robot"),
    #     },
    # )


@configclass
class SceneRewardsCfg:
    """Reward terms for the MDP."""

    pass


@configclass
class SceneTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=orbit_mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=orbit_mdp.base_height, params={"minimum_height": 0.00, "asset_cfg": SceneEntityCfg("object")}
    )

    rope_invalid_state = DoneTerm(func=tycho_general_mdp.invalid_state, params={"asset_cfg": SceneEntityCfg("object")})
