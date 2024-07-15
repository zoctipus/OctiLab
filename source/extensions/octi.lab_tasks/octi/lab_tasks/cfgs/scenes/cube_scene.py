from __future__ import annotations
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import AssetBaseCfg, RigidObjectCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
import omni.isaac.lab.envs.mdp as orbit_mdp
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip
from octi.lab.sim.spawners.from_files import MultiAssetCfg

RADIUS = 0.02

marker_cfg = FRAME_MARKER_CFG.copy()
marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
marker_cfg.prim_path = "/Visuals/RopeFrameTransformer"


@configclass
class SceneObjectSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene"""

    # object: AssetBaseCfg = RigidObjectCfg(
    #         prim_path="{ENV_REGEX_NS}/Object",
    #         init_state=RigidObjectCfg.InitialStateCfg(pos= [0, 0, 0.06], rot=[1, 0, 0, 0]),
    #         spawn=UsdFileCfg(
    #             usd_path="assets/ball.usd",
    #             #true radius is actually RADIUS / 2.
    #             scale=(RADIUS*2, RADIUS*2, RADIUS*2),
    #         ),
    #     )
    # Due to current randomization does not randomize scale of object, the object radius needs to be
    # a bit smaller than the ball's in order for the chopsticks reward function to work correctly
    # you can see that 0.015cm instead of RADIUS is used for scale of cube
    # This is something that needs to be fixed in future

    # object: AssetBaseCfg = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Object",
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 0, 0.016), rot=(1, 0, 0, 0)),
    #     spawn=UsdFileCfg(
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
    #         scale=(0.015/0.02, 0.015/0.02, 0.015/0.02),
    #         rigid_props=RigidBodyPropertiesCfg(
    #             solver_position_iteration_count=16,
    #             solver_velocity_iteration_count=1,
    #             max_angular_velocity=1000.0,
    #             max_linear_velocity=1000.0,
    #             max_depenetration_velocity=5.0,
    #             disable_gravity=False,
    #         ),
    #     ),
    # )

    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=MultiAssetCfg(
            assets_cfg=[
                sim_utils.CuboidCfg(
                    size=(0.1, 0.13, 0.13),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                ),
                sim_utils.CuboidCfg(
                    size=(0.13, 0.1, 0.13),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                ),
                sim_utils.CuboidCfg(
                    size=(0.1, 0.1, 0.1),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                ),
                sim_utils.SphereCfg(
                    radius=0.1,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
                ),
            ],
            scaling_range={'x': (0.4, 0.8), 'y': (0.4, 0.8), 'z': (0.4, 0.8)},
            uniform_scale=True
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, 0.0, 0.03)),
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
class SceneEventCfg:
    """Configuration for randomization."""
    # pass
    reset_object_position = EventTerm(
        func=orbit_mdp.reset_root_state_with_random_orientation,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.3), "y": (-0.4, 0.4), "z": (0.00, 0.00)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )

    # rescale_object_size = EventTerm(
    #     func=orbit_mdp.randomize_rigid_body_material,
    #     mode="reset",
    #     params={
    #         "static_friction_range": [0.25, 1.0],
    #         "dynamic_friction_range": [0.25, 1.0],
    #         "restitution_range": [0.0, 0.5],
    #         "num_buckets": 1,
    #         "asset_cfg": SceneEntityCfg("object", body_names="Object"),
    #     }
    # )

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

    add_rigid_body_mass = EventTerm(
        func=orbit_mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

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
        func=orbit_mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )
