"""Configuration for the Hebi robots.

The following configurations are available:

* :obj:`HEBI_CFG`: Hebi robot with chopsticks
"""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg, IdealPDActuatorCfg
from octilab.actuators import HebiMotorCfg

from omni.isaac.lab.assets.articulation import ArticulationCfg
from octilab.assets.articulation import HebiArticulationCfg

from omni.isaac.lab.sensors import TiledCameraCfg, FrameTransformerCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip

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


HEBI_DEFAULT_JOINTPOS = {
    'HEBI_base_X8_9': -2.2683857389667805,
    'HEBI_shoulder_X8_16': 1.5267610481188283,
    'HEBI_elbow_X8_9': 2.115358222505881,
    'HEBI_wrist1_X5_1': 0.5894993521468314,
    'HEBI_wrist2_X5_1': 0.8740650991816328,
    'HEBI_wrist3_X5_1': 0.0014332898815118368,
    'HEBI_chopstick_X5_1': -0.36,
}

HEBI_ORBIT_ARTICULATION = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="datasets/tycho_robot.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=1, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos=HEBI_DEFAULT_JOINTPOS
    ),
    soft_joint_pos_limit_factor=1,
)

HEBI_CUSTOM_ARTICULATION = HebiArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="datasets/tycho_robot.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=1, solver_velocity_iteration_count=0
        ),
    ),
    init_state=HebiArticulationCfg.InitialStateCfg(
        joint_pos=HEBI_DEFAULT_JOINTPOS
    ),
    soft_joint_pos_limit_factor=1
)

ORIGIN_HEBI_ARTICULATION = HebiArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="datasets/hebi_origin.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=1, solver_velocity_iteration_count=0
        ),
    ),
    init_state=HebiArticulationCfg.InitialStateCfg(
        joint_pos={'HEBI_base_X8_9': -2.2683857389667805,
                   'HEBI_shoulder_X8_16': 1.5267610481188283,
                   'HEBI_elbow_X8_9': 2.115358222505881,
                   'HEBI_wrist1_X5_1': 0.5894993521468314,
                   'HEBI_wrist2_X5_1': 0.8740650991816328,
                   'HEBI_wrist3_X5_1': 0.0014332898815118368,
                   'HEBI_chopstick_actuator_X5_1': -0.36}
    ),
    soft_joint_pos_limit_factor=1
)

origin_implicit_stiffness_scalar = 50
origin_implicit_damping_scalar = 10
ORIGIN_HEBI_IMPLICITY_ACTUATOR_CFG = ORIGIN_HEBI_ARTICULATION.copy()
ORIGIN_HEBI_IMPLICITY_ACTUATOR_CFG.actuators = {
    "X8_9": ImplicitActuatorCfg(
        joint_names_expr=["HEBI_base_X8_9", "HEBI_elbow_X8_9"],
        stiffness=3.0 * origin_implicit_stiffness_scalar,
        damping=2. * origin_implicit_damping_scalar,
        armature=0.001,
        friction=0.2,
        velocity_limit=1,
        effort_limit=500,
    ),
    "X8_16": ImplicitActuatorCfg(
        joint_names_expr=["HEBI_shoulder_X8_16"],
        stiffness=3.0 * origin_implicit_stiffness_scalar,
        damping=2. * origin_implicit_damping_scalar,
        armature=0.001,
        friction=0.2,
        velocity_limit=1,
        effort_limit=500,
    ),
    "x5": ImplicitActuatorCfg(
        joint_names_expr=["HEBI_wrist1_X5_1", "HEBI_wrist2_X5_1", "HEBI_wrist3_X5_1"],
        stiffness=1.0 * origin_implicit_stiffness_scalar,
        damping=0.3 * origin_implicit_damping_scalar,
        armature=0.001,
        friction=0.2,
        velocity_limit=1,
        effort_limit=500,
    ),
    "chop": ImplicitActuatorCfg(
        joint_names_expr=["HEBI_chopstick_actuator_X5_1"],
        stiffness=1.0 * origin_implicit_stiffness_scalar / 5,
        damping=0.3 * origin_implicit_damping_scalar / 5,
        armature=0.001,
        friction=0.2,
        velocity_limit=1,
        effort_limit=500,
    ),
}


implicit_stiffness_scalar = 40
implicit_damping_scalar = 10
HEBI_IMPLICITY_ACTUATOR_CFG = HEBI_ORBIT_ARTICULATION.copy()
HEBI_IMPLICITY_ACTUATOR_CFG.actuators = {
    "X8_9": ImplicitActuatorCfg(
        joint_names_expr=["HEBI_base_X8_9", "HEBI_elbow_X8_9"],
        stiffness=3.0 * implicit_stiffness_scalar,
        damping=2. * implicit_damping_scalar,
        armature=0.001,
        friction=0.2,
        velocity_limit=1,
        effort_limit=500,
    ),
    "X8_16": ImplicitActuatorCfg(
        joint_names_expr=["HEBI_shoulder_X8_16"],
        stiffness=3.0 * implicit_stiffness_scalar,
        damping=2. * implicit_damping_scalar,
        armature=0.001,
        friction=0.2,
        velocity_limit=1,
        effort_limit=500,
    ),
    "x5": ImplicitActuatorCfg(
        joint_names_expr=["HEBI_wrist1_X5_1", "HEBI_wrist2_X5_1", "HEBI_wrist3_X5_1"],
        stiffness=1.0 * implicit_stiffness_scalar,
        damping=0.3 * implicit_damping_scalar,
        armature=0.001,
        friction=0.2,
        velocity_limit=1,
        effort_limit=500,
    ),
    "chop": ImplicitActuatorCfg(
        joint_names_expr=["HEBI_chopstick_X5_1"],
        stiffness=1.0 * implicit_stiffness_scalar / 5,
        damping=0.3 * implicit_damping_scalar / 5,
        armature=0.001,
        friction=0.2,
        velocity_limit=1,
        effort_limit=500,
    ),
}


maxtorch = [23.3, 44.7632, 23.3, 2.66, 2.66, 2.66, 2.66]
speed_24v = [4.4843, 2.3375, 4.4843, 14.12, 14.12, 14.12, 14.12]
HEBI_PWM_MOTOR_CFG = HEBI_CUSTOM_ARTICULATION.copy()
HEBI_PWM_MOTOR_CFG.actuators = {
    "HEBI": HebiMotorCfg(
        joint_names_expr=["HEBI_base_X8_9", "HEBI_elbow_X8_9", "HEBI_shoulder_X8_16",
                          "HEBI_wrist1_X5_1", "HEBI_wrist2_X5_1", "HEBI_wrist3_X5_1", "HEBI_chopstick_X5_1"],
        stiffness=100.0,
        damping=10,
        armature=0.001,
        friction=0.2,
        velocity_limit=50,
        effort_limit=1000,
        actuator_biasprm=[maxtorch, speed_24v],
        gain_xml_path="datasets/chopstick-gains-7D-all3.xml",
        only_position_control=False,
    ),
    "HEBI_chopstick_X5_1": ImplicitActuatorCfg(
        joint_names_expr=["HEBI_chopstick_X5_1"],
        stiffness=50,
        damping=10,
        armature=0.01,
        friction=0.2,
        velocity_limit=50,
        effort_limit=10000
    )
}

pd_stiffness_scalar = 50
pd_damping_scalar = 50
HEBI_IDEAL_PD_CFG = HEBI_ORBIT_ARTICULATION.copy()
HEBI_IDEAL_PD_CFG.actuators = {
    "X8_9": IdealPDActuatorCfg(
        joint_names_expr=["HEBI_base_X8_9", "HEBI_elbow_X8_9"],
        stiffness=3.0 * pd_stiffness_scalar,
        damping=0.025 * pd_damping_scalar,
        armature=0.001,
        friction=0.2,
        velocity_limit=50,
        effort_limit=10000
    ),

    "X8_16": IdealPDActuatorCfg(
        joint_names_expr=["HEBI_shoulder_X8_16"],
        stiffness=7.0 * pd_stiffness_scalar,
        damping=0.03 * pd_damping_scalar,
        armature=0.001,
        friction=0.2,
        velocity_limit=50,
        effort_limit=10000
    ),
    "X5_1": IdealPDActuatorCfg(
        joint_names_expr=["HEBI_wrist1_X5_1", "HEBI_wrist2_X5_1", "HEBI_wrist3_X5_1"],
        stiffness=2.0 * pd_stiffness_scalar,
        damping=0.10 * pd_damping_scalar,
        armature=0.001,
        friction=0.2,
        velocity_limit=50,
        effort_limit=10000
    ),
    "HEBI_chopstick_X5_1": ImplicitActuatorCfg(
        joint_names_expr=["HEBI_chopstick_X5_1"],
        stiffness=1 * pd_stiffness_scalar,
        damping=0.1 * pd_damping_scalar,
        armature=0.01,
        friction=0.2,
        velocity_limit=50,
        effort_limit=10000
    )
}

all3_stiffness_scalar = 30
all3_damping_scalar = 10
HEBI_ALL3_CFG = HEBI_ORBIT_ARTICULATION.copy()
HEBI_ALL3_CFG.actuators = {
    "HEBI_base_X8_9": IdealPDActuatorCfg(
        joint_names_expr=["HEBI_base_X8_9"],
        stiffness=3.0 * all3_stiffness_scalar,
        damping=0.5 * all3_damping_scalar,
        armature=0.001,
        friction=0.2,
        velocity_limit=50,
        effort_limit=10000
    ),
    "HEBI_elbow_X8_9": IdealPDActuatorCfg(
        joint_names_expr=["HEBI_elbow_X8_9"],
        stiffness=5.5 * all3_stiffness_scalar,
        damping=0.5 * all3_damping_scalar,
        armature=0.001,
        friction=0.2,
        velocity_limit=50,
        effort_limit=10000
    ),
    "HEBI_shoulder_X8_16": IdealPDActuatorCfg(
        joint_names_expr=["HEBI_shoulder_X8_16"],
        stiffness=7.0 * all3_stiffness_scalar,
        damping=1.2 * all3_damping_scalar,
        armature=0.001,
        friction=0.2,
        velocity_limit=50,
        effort_limit=10000
    ),
    "HEBI_wrist1_X5_1": IdealPDActuatorCfg(
        joint_names_expr=["HEBI_wrist1_X5_1"],
        stiffness=7.0 * all3_stiffness_scalar,
        damping=0.1 * all3_damping_scalar,
        armature=0.001,
        friction=0.2,
        velocity_limit=50,
        effort_limit=10000
    ),
    "HEBI_wrist2_X5_1": IdealPDActuatorCfg(
        joint_names_expr=["HEBI_wrist2_X5_1"],
        stiffness=2.0 * all3_stiffness_scalar,
        damping=0.01 * all3_damping_scalar,
        armature=0.1,
        friction=0.2,
        velocity_limit=50,
        effort_limit=10000
    ),
    "HEBI_wrist3_X5_1": IdealPDActuatorCfg(
        joint_names_expr=["HEBI_wrist3_X5_1"],
        stiffness=3.5 * all3_stiffness_scalar,
        damping=0.1 * all3_damping_scalar,
        armature=0.001,
        friction=0.2,
        velocity_limit=50,
        effort_limit=10000
    ),
    "HEBI_chopstick_X5_1": IdealPDActuatorCfg(
        joint_names_expr=["HEBI_chopstick_X5_1"],
        stiffness=1 * all3_stiffness_scalar,
        damping=0.1 * all3_damping_scalar,
        armature=0.01,
        friction=0.2,
        velocity_limit=50,
        effort_limit=10000
    )
}


'''
FRAMES
'''
marker_cfg = FRAME_MARKER_CFG.copy()
marker_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
marker_cfg.prim_path = "/Visuals/FrameTransformer"

FRAME_EE = FrameTransformerCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base_shoulder",
    debug_vis=False,
    visualizer_cfg=marker_cfg,
    target_frames=[
        FrameTransformerCfg.FrameCfg(
            prim_path="{ENV_REGEX_NS}/Robot/static_chop_tip",
            name="ee",
            offset=OffsetCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0)
                # rot=(0.7070904, -0.7071232, 0, 0)
            ),
        ),
    ],
)

f1 = FrameTransformerCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base_shoulder",
    debug_vis=True,
    visualizer_cfg=marker_cfg,
    target_frames=[
        FrameTransformerCfg.FrameCfg(
            prim_path="{ENV_REGEX_NS}/Robot/shoulder_elbow",
            name="1",
        ),
    ],
)

f2 = FrameTransformerCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base_shoulder",
    debug_vis=True,
    visualizer_cfg=marker_cfg,
    target_frames=[
        FrameTransformerCfg.FrameCfg(
            prim_path="{ENV_REGEX_NS}/Robot/elbow_wrist1",
            name="2",
        ),
    ],
)

f3 = FrameTransformerCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base_shoulder",
    debug_vis=True,
    visualizer_cfg=marker_cfg,
    target_frames=[
        FrameTransformerCfg.FrameCfg(
            prim_path="{ENV_REGEX_NS}/Robot/wrist1_wrist2",
            name="3",
        ),
    ],
)

f4 = FrameTransformerCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base_shoulder",
    debug_vis=True,
    visualizer_cfg=marker_cfg,
    target_frames=[
        FrameTransformerCfg.FrameCfg(
            prim_path="{ENV_REGEX_NS}/Robot/wrist2_wrist3",
            name="4",
        ),
    ],
)

f5 = FrameTransformerCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base_shoulder",
    debug_vis=True,
    visualizer_cfg=marker_cfg,
    target_frames=[
        FrameTransformerCfg.FrameCfg(
            prim_path="{ENV_REGEX_NS}/Robot/wrist3_chopstick",
            name="1",
        ),
    ],
)

f6 = FrameTransformerCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base_shoulder",
    debug_vis=True,
    visualizer_cfg=marker_cfg,
    target_frames=[
        FrameTransformerCfg.FrameCfg(
            prim_path="{ENV_REGEX_NS}/Robot/end_effector",
            name="5",
        ),
    ],
)

FRAME_FIXED_CHOP_TIP = FrameTransformerCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base_shoulder",
    debug_vis=False,
    visualizer_cfg=marker_cfg,
    target_frames=[
        FrameTransformerCfg.FrameCfg(
            prim_path="{ENV_REGEX_NS}/Robot/wrist3_chopstick",
            name="fixed_chop_tip",
            offset=OffsetCfg(
                pos=(0.13018, 0.07598, 0.06429),
            ),
        ),
    ],
)

FRAME_FIXED_CHOP_END = FrameTransformerCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base_shoulder",
    debug_vis=False,
    visualizer_cfg=marker_cfg,
    target_frames=[
        FrameTransformerCfg.FrameCfg(
            prim_path="{ENV_REGEX_NS}/Robot/wrist3_chopstick",
            name="fixed_chop_end",
            offset=OffsetCfg(
                pos=(-0.13134, 0.07598, 0.06424),
            ),
        ),
    ],
)

FRAME_FREE_CHOP_TIP = FrameTransformerCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base_shoulder",
    debug_vis=False,
    visualizer_cfg=marker_cfg,
    target_frames=[
        FrameTransformerCfg.FrameCfg(
            prim_path="{ENV_REGEX_NS}/Robot/end_effector",
            name="free_chop_tip",
            offset=OffsetCfg(
                pos=(0.12001, 0.05445, 0.00229),
            ),
        ),
    ],
)

FRAME_FREE_CHOP_END = FrameTransformerCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base_shoulder",
    debug_vis=False,
    visualizer_cfg=marker_cfg,
    target_frames=[
        FrameTransformerCfg.FrameCfg(
            prim_path="{ENV_REGEX_NS}/Robot/end_effector",
            name="free_chop_end",
            offset=OffsetCfg(
                pos=(-0.11378, -0.04546, 0.00231),
            ),
        ),
    ],
)

CAMERA_WRIST = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/wrist3_chopstick/camera_mount/CameraSensor",
    update_period=0,
    offset=TiledCameraCfg.OffsetCfg(
        pos=(-0.05, 0.02, 0.06),
        rot=(0.0, 0.70711, 0.0, -0.70711),
        convention="opengl"
    ),
    data_types=["rgb",],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20)
    ),
    width=80,
    height=80,
)

CAMERA_BASE = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base_shoulder/camera_mount/CameraSensor",
    update_period=0,
    offset=TiledCameraCfg.OffsetCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(0.81915, 0, 0.57358, 0),
        convention="opengl"
    ),
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
    ),
    width=80,
    height=80,
)
