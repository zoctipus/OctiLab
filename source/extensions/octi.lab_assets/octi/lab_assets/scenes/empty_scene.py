from __future__ import annotations
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import AssetBaseCfg
from omni.isaac.lab.assets import AssetBaseCfg, RigidObjectCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
import omni.isaac.lab.envs.mdp as orbit_mdp
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip

RADIUS = 0.02

marker_cfg = FRAME_MARKER_CFG.copy()
marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
marker_cfg.prim_path = "/Visuals/RopeFrameTransformer"


@configclass
class SceneObjectSceneCfg(InteractiveSceneCfg):
    """Configuration for an empty scene"""
    # ground plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, 0)),
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
    pass


@configclass
class SceneRewardsCfg:
    """Reward terms for the MDP."""
    pass


@configclass
class SceneTerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=orbit_mdp.time_out, time_out=True)
