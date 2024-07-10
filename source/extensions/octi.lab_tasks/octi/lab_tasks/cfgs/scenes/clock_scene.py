from __future__ import annotations
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
import omni.isaac.lab.envs.mdp as orbit_mdp
import octi.lab.envs.mdp as general_mdp

from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip

RADIUS = 0.02

marker_cfg = FRAME_MARKER_CFG.copy()  # type: ignore
marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
marker_cfg.prim_path = "/Visuals/RopeFrameTransformer"


@configclass
class SceneObjectSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene"""

    object: AssetBaseCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path="datasets/articulation_objects/clock.usd",
            scale=(0.1, 0.1, 0.1),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=16, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.4, -0.22, 0.00),
            rot=(0.0, 0, 0, 1.0),
            joint_pos={
                "joint_1": 0.0,
            },
        ),
        actuators={"hands": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.1, damping=0.1)},
    )

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.62305, 0, 0), rot=(0.707, 0, 0, -0.707)),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -1.05)),
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
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
        func=orbit_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "z": (0.05, 0.05)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )

    update_joint_position_target = EventTerm(
        func=general_mdp.update_joint_target_positions_to_current,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={"asset_name": "object"},
    )


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
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    )
