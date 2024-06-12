
from __future__ import annotations

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import AssetBaseCfg, RigidObjectCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
import omni.isaac.lab.envs.mdp as orbit_mdp

from octilab.assets import DeformableCfg


@configclass
class SceneObjectSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene"""
    wooden_board: AssetBaseCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/wooden_board",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.3308790688293063, -0.459486269649666, -3.4006016153006077e-7),
            rot=(0.70711, 0.0, 0.0, 0.70711)),
        spawn=UsdFileCfg(
            usd_path="datasets/rigidbodies/wooden_board.usda",
            scale=(0.5, 0.5, 0.5),
        ),
    )

    wagyu: AssetBaseCfg = DeformableCfg(
        prim_path="{ENV_REGEX_NS}/wagyu",
        init_state=DeformableCfg.InitialStateCfg(pos=(-0.3128404333733237, -0.45898978548773034, 0.013707292356420923),
                                                 rot=(0.6694, 0.6694, -0.22783, -0.22783)),
        spawn=UsdFileCfg(
            usd_path="datasets/deformables/wagyu.usd",
            scale=(0.9, 0.9, 0.9),
        ),
    )

    induction_stove: AssetBaseCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/induction_stove",
        init_state=RigidObjectCfg.InitialStateCfg
        (
            pos=(-0.5130457698840921, -0.2461988265834078, 0.013625786573623079),
            rot=(0.49026, 0.49025, -0.50956, -0.50956)
        ),
        spawn=UsdFileCfg(
            usd_path="datasets/rigidbodies/induction_cook.usd",
            scale=(1.8, 1.8, 1.8),
        ),
    )

    cast_iron_skillet: AssetBaseCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cast_iron_skillet",
        init_state=RigidObjectCfg.InitialStateCfg
        (
            pos=(-0.5131438058661296, -0.3138202611064712, 0.02727337380958563),
            rot=(1, 0.0, 0.0, 0.0)
        ),
        spawn=UsdFileCfg(
            usd_path="datasets/rigidbodies/cast_iron_skillet.usd",
            scale=(0.9, 0.9, 0.9),
        ),
    )

    onion_section: AssetBaseCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/onion_section",
        init_state=RigidObjectCfg.InitialStateCfg
        (
            pos=(-0.27793083362515747, -0.538901170056425, 0.01254465398449706),
            rot=(0.70704, 0.70704, -0.00982, -0.00982)
        ),
        spawn=UsdFileCfg(
            usd_path="datasets/rigidbodies/onion_section.usd",
            scale=(0.5, 0.4, 0.5),
        ),
    )

    rosemary1: AssetBaseCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/rosemary1",
        init_state=RigidObjectCfg.InitialStateCfg
        (
            pos=(-0.3046810561276882, -0.5766609175208751, 0.01000015711686957),
            rot=(0.6594, 0.6594, 0.25534, 0.25534)
        ),
        spawn=UsdFileCfg(
            usd_path="datasets/rigidbodies/rosemary1.usd",
            scale=(1.0, 1.0, 1.0),
        ),
    )

    rosemary2: AssetBaseCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/rosemary2",
        init_state=RigidObjectCfg.InitialStateCfg
        (
            pos=(-0.33098440387155104, -0.5688773665720779, 0.018113191730254784),
            rot=(0.70349, 0.62552, -0.32974, -0.0714)
        ),
        spawn=UsdFileCfg(
            usd_path="datasets/rigidbodies/rosemary2.usd",
            scale=(1.0, 1.0, 1.0),
        ),
    )

    table: AssetBaseCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/dinning_table",
        init_state=RigidObjectCfg.InitialStateCfg
        (
            pos=(-0.24571, -0.01515, -0.77),
            rot=(0.70711, 0.0, 0.0, -0.70711)
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


@configclass
class SceneCommandsCfg:
    """Command terms for the Scene."""
    pass


@configclass
class SceneEventCfg:
    """Configuration for randomization."""
    # pass
    reset_meat_position = EventTerm(
        func=orbit_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("wagyu", body_names="wagyu"),
        },
    )

    reset_cast_iron_skillet_position = EventTerm(
        func=orbit_mdp.reset_root_state_uniform2,
        mode="reset",
        params={
            "pose_range": {},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cast_iron_skillet", body_names="cast_iron_skillet"),
        },
    )

    reset_wooden_board_position = EventTerm(
        func=orbit_mdp.reset_root_state_uniform2,
        mode="reset",
        params={
            "pose_range": {},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("wooden_board", body_names="wooden_board"),
        },
    )

    reset_induction_stove_position = EventTerm(
        func=orbit_mdp.reset_root_state_uniform2,
        mode="reset",
        params={
            "pose_range": {},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("induction_stove", body_names="induction_stove"),
        },
    )

    reset_rosemary2_position = EventTerm(
        func=orbit_mdp.reset_root_state_uniform2,
        mode="reset",
        params={
            "pose_range": {},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("rosemary2", body_names="rosemary2"),
        },
    )

    reset_rosemary1_position = EventTerm(
        func=orbit_mdp.reset_root_state_uniform2,
        mode="reset",
        params={
            "pose_range": {},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("rosemary1", body_names="rosemary1"),
        },
    )

    reset_onion_section_position = EventTerm(
        func=orbit_mdp.reset_root_state_uniform2,
        mode="reset",
        params={
            "pose_range": {},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("onion_section", body_names="onion_section"),
        },
    )


@configclass
class SceneRewardsCfg:
    """Reward terms for the MDP."""
    pass


@configclass
class SceneTerminationsCfg:
    """Termination terms for the MDP."""
    pass
