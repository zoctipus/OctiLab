"""Configuration for the Hebi robots.

The following configurations are available:

* :obj:`HEBI_CFG`: Hebi robot with chopsticks
"""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg

from omni.isaac.lab.assets.articulation import ArticulationCfg

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


LEAP_DEFAULT_JOINT_POS = {
    "w0": 0.0,
    "w1": 0.0,
    "w2": 0.0,
    "w3": 0.0,
    "w4": 0.0,
    "w5": 0.0,
    "j1": 0.0,
    "j2": 0.0,
    "j3": 0.0,
    "j4": 0.0,
    "j5": 0.0,
    "j6": 0.0,
    "j7": 0.0,
    "j8": 0.0,
    "j9": 0.0,
    "j10": 0.0,
    "j11": 0.0,
    "j12": 0.0,
    "j13": 0.0,
    "j14": 0.0,
    "j15": 0.0,
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


implicit_stiffness_scalar = 40
implicit_damping_scalar = 10
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
