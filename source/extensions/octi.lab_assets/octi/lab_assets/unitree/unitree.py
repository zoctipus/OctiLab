# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Octilab Extension Configuration for Unitree robots.

While the following configurations are available:

* :obj:`UNITREE_A1_CFG`: Unitree A1 robot with DC motor model for the legs
* :obj:`UNITREE_GO1_CFG`: Unitree Go1 robot with actuator net model for the legs
* :obj:`UNITREE_GO2_CFG`: Unitree Go2 robot with DC motor model for the legs
* :obj:`H1_CFG`: H1 humanoid robot
* :obj:`G1_CFG`: G1 humanoid robot

Reference: https://github.com/unitreerobotics/unitree_ros

We provide Additional:
* :obj:`UNITREE_A1_IMPLICIT_ACTUATOR_CFG`: Unitree A1 robot with implicit motor model for the legs
"""

from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab_assets.unitree import UNITREE_A1_CFG

##
# Configuration - Actuators.
##


"""Configuration of A1 actuators using Implicit model.

Actuator specifications: https://shop.unitree.com/products/go1-motor

This model is taken from: https://github.com/Improbable-AI/walk-these-ways
"""

##
# Configuration
##

UNITREE_A1_IMPLICIT_ACTUATOR_CFG = UNITREE_A1_CFG.replace(
    actuators={"base_legs": ImplicitActuatorCfg(
        joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        effort_limit=33.5,
        velocity_limit=21.0,
        stiffness=25.0,
        damping=0.5,
    )})
