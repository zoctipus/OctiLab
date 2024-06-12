
from __future__ import annotations
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import AssetBaseCfg, RigidObjectCfg
from assets import DeformableCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
import omni.isaac.lab.envs.mdp as orbit_mdp


@configclass
class SceneObjectSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene"""

    wagyu: AssetBaseCfg = DeformableCfg(
        prim_path="{ENV_REGEX_NS}/wagyu",
        init_state=DeformableCfg.InitialStateCfg(pos=(-0.20, -0.2, 0.05), rot=(1, 0, 0, 0)),
        spawn=UsdFileCfg(
            usd_path="datasets/deformables/wagyu.usd",
            scale=(1, 1, 1),
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
            "pose_range":
            {
                "x": (0, 0),
                "y": (0, 0),
                "z": (0, 0)
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("wagyu", body_names="wagyu"),
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
