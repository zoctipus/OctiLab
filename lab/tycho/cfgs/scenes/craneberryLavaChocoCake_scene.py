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
import omni.isaac.lab.envs.mdp as orbit_mdp

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
class SceneObjectSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene"""

    cake: AssetBaseCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cake",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(cake_position[0], cake_position[1], cake_position[2]), rot=(1, 0, 0, 0)),
        spawn=UsdFileCfg(
            usd_path="datasets/rigidbodies/cake_on_plate.usd",
            scale=(cake_scale, cake_scale, cake_scale),
        ),
    )

    # plate: AssetBaseCfg = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/plate",
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #           pos=(plate_position[0], plate_position[1], plate_position[2]), rot=(1, 0, 0, 0)),
    #     spawn=UsdFileCfg(
    #         usd_path=f"datasets/rigidbodies/white_plate_2.usd",
    #         scale=(plate_scale, plate_scale, plate_scale),
    #     ),
    # )

    canberry: AssetBaseCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/canberry",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(canberry_position[0], canberry_position[1], cake_position[2]), rot=(1, 0, 0, 0)),
        spawn=UsdFileCfg(
            usd_path="datasets/rigidbodies/canberry.usd",
            scale=(canberry_scale, canberry_scale, canberry_scale),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=16,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )

    P_Glassware_Short: AssetBaseCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/P_Glassware_Short",
        init_state=RigidObjectCfg.InitialStateCfg
        (
            pos=(p_glassware_short_position[0], p_glassware_short_position[1], p_glassware_short_position[2]),
            rot=(1, 0.0, 0.0, 0.0)
        ),
        spawn=UsdFileCfg(
            usd_path="datasets/rigidbodies/P_Glassware_Short.usd",
            scale=(p_glassware_short_scale, p_glassware_short_scale, p_glassware_short_scale),
        ),
    )

    spoon: AssetBaseCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/spoon",
        init_state=RigidObjectCfg.InitialStateCfg
        (
            pos=(spoon_position[0], spoon_position[1], spoon_position[2]),
            rot=(0.71766, 0.6964, 0.0, 0.0)
        ),
        spawn=UsdFileCfg(
            usd_path="datasets/rigidbodies/spoon.usd",
            scale=(spoon_scale, spoon_scale, spoon_scale),
        ),
    )

    table: AssetBaseCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/dinning_table",
        init_state=RigidObjectCfg.InitialStateCfg
        (
            pos=(-0.35, -0.25, -0.765),
            rot=(1, 0.0, 0.0, 0.0)
        ),
        spawn=UsdFileCfg(
            usd_path="datasets/rigidbodies/DesPeres_Table.usd",
            scale=(1, 1, 1),
        ),
    )

    # ground plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -0.765)),
        spawn=sim_utils.GroundPlaneCfg(),
    )

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
    reset_cake_position = EventTerm(
        func=orbit_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range":
            {
                "x": (0, 0),
                "y": (0, 0),
                "z": (0, 0)
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cake", body_names="cake"),
        },
    )

    reset_caneberry_position = EventTerm(
        func=orbit_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.10, 0.10),
                           "y": (-0.1, 0.1),
                           "z": (0, 0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("canberry", body_names="canberry"),
        },
    )
    # reset_plate_position = EventTerm(
    #     func=orbit_mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range":
    #         {
    #             "x": (-0.35, -0.35),
    #             "y": (-0.15, -0.15),
    #             "z": (0, 0)
    #         },
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("plate", body_names="plate"),
    #     },
    # )

    reset_P_Glassware_Short_position = EventTerm(
        func=orbit_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("P_Glassware_Short", body_names="P_Glassware_Short"),
        },
    )

    spoon = EventTerm(
        func=orbit_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("spoon", body_names="spoon"),
        },
    )

    # reset_caneberryTree_position = EventTerm(
    #     func=orbit_mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range":
    #         {
    #             "x":(canberryTree_position[0], canberryTree_position[0]),
    #             "y":(canberryTree_position[1], canberryTree_position[1]),
    #             "z":(canberryTree_position[2], canberryTree_position[2])
    #         },
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("canberry_tree", body_names="canberry_tree"),
    #     },
    # )


@configclass
class SceneRewardsCfg:
    """Reward terms for the MDP."""
    pass


@configclass
class SceneTerminationsCfg:
    """Termination terms for the MDP."""
    cake_dropping = DoneTerm(
        func=orbit_mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cake")}
    )

    berry_dropping = DoneTerm(
        func=orbit_mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("canberry")}
    )
