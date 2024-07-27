"""Configuration for the Hebi robots.

The following configurations are available:

* :obj:`HEBI_CFG`: Hebi robot with chopsticks
"""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg

from omni.isaac.lab.sensors import FrameTransformerCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
##
# Configuration
##

"""
HINTS FROM ISAAC SIM DOCUMENTATION
https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.core/docs/index.html#articulationview

High stiffness makes the joints snap faster and harder to the desired target,
and higher damping smoothes but also slows down the joint's movement to target

For position control, set relatively high stiffness and low damping (to reduce vibrations)

For velocity control, stiffness must be set to zero with a non-zero damping

For effort control, stiffness and damping must be set to zero
"""

"""
LEAP HAND
"""
LEAP_DEFAULT_JOINT_POS = {
    "w0": 0.0, "w1": 0.0, "w2": 0.0, "w3": 0.0,
    "w4": 0.0, "w5": 0.0, "j1": 0.0, "j2": 0.0,
    "j3": 0.0, "j4": 0.0, "j5": 0.0, "j6": 0.0,
    "j7": 0.0, "j8": 0.0, "j9": 0.0, "j10": 0.0,
    "j11": 0.0, "j12": 0.0, "j13": 0.0, "j14": 0.0, "j15": 0.0,
}

LEAP_ARTICULATION = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="datasets/robots/leap_hand/leap_hand_6d.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=1, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0, 0, 0), rot=(1, 0, 0, 0), joint_pos=LEAP_DEFAULT_JOINT_POS),
    soft_joint_pos_limit_factor=1,
)


IMPLICIT_LEAP = LEAP_ARTICULATION.copy()  # type: ignore
IMPLICIT_LEAP.actuators = {
    "w": ImplicitActuatorCfg(
        joint_names_expr=["w.*"],
        stiffness=200.0,
        damping=50.0,
        armature=0.001,
        friction=0.2,
        velocity_limit=1,
        effort_limit=50,
    ),
    "j": ImplicitActuatorCfg(
        joint_names_expr=["j.*"],
        stiffness=200.0,
        damping=30.0,
        armature=0.001,
        friction=0.2,
        velocity_limit=1,
        effort_limit=50,
    ),
}

"""
LEAP_XARM
"""

LEAP_XARM_DEFAULT_JOINT_POS = {
    "j1": 0.0, "j2": 0.0,
    "j3": 0.0, "j4": 0.0, "j5": 0.0, "j6": 0.0,
    "j7": 0.0, "j8": 0.0, "j9": 0.0, "j10": 0.0,
    "j11": 0.0, "j12": 0.0, "j13": 0.0, "j14": 0.0, "j15": 0.0,
    "a2": 0.0, "a2": 0.0, "a3": 0.0, "a4": 0.0, "a5": 0.0,
}

LEAP_XARM_ARTICULATION = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="datasets/robots/xarm/leap_xarm.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=1, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0, 0, 0), rot=(1, 0, 0, 0), joint_pos=LEAP_XARM_DEFAULT_JOINT_POS),
    soft_joint_pos_limit_factor=1,
)

IMPLICIT_LEAP_XARM = LEAP_XARM_ARTICULATION.copy()  # type: ignore
IMPLICIT_LEAP_XARM.actuators = {
    "a": ImplicitActuatorCfg(
        joint_names_expr=["a.*"],
        stiffness=200.0,
        damping=50.0,
        armature=0.001,
        friction=0.2,
        velocity_limit=1,
        effort_limit=50,
    ),
    "j": ImplicitActuatorCfg(
        joint_names_expr=["j.*"],
        stiffness=300.0,
        damping=10.0,
        armature=0.001,
        friction=0.2,
        velocity_limit=1,
        effort_limit=5,
    ),
}


"""
FRAMES
"""
marker_cfg = FRAME_MARKER_CFG.copy()  # type: ignore
marker_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
marker_cfg.prim_path = "/Visuals/FrameTransformer"

FRAME_EE = FrameTransformerCfg(
    prim_path="{ENV_REGEX_NS}/Robot/link_base",
    debug_vis=False,
    visualizer_cfg=marker_cfg,
    target_frames=[
        FrameTransformerCfg.FrameCfg(
            prim_path="{ENV_REGEX_NS}/Robot/palm_lower",
            name="ee",
            offset=OffsetCfg(
                pos=(-0.028, -0.04, -0.07),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        ),
    ],
)
