# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the ANYbotics robots.

The following configuration parameters are available:

* :obj:`ANYMAL_B_CFG`: The ANYmal-B robot with ANYdrives 3.0
* :obj:`ANYMAL_C_CFG`: The ANYmal-C robot with ANYdrives 3.0
* :obj:`ANYMAL_D_CFG`: The ANYmal-D robot with ANYdrives 3.0

Reference:

* https://github.com/ANYbotics/anymal_b_simple_description
* https://github.com/ANYbotics/anymal_c_simple_description
* https://github.com/ANYbotics/anymal_d_simple_description

"""

from omni.isaac.lab.actuators import ImplicitActuatorCfg
from octi.lab.actuators import EffortMotorCfg
from omni.isaac.lab.sensors import RayCasterCfg

from omni.isaac.lab_assets.velodyne import VELODYNE_VLP_16_RAYCASTER_CFG

from omni.isaac.lab_assets.anymal import (ANYDRIVE_3_SIMPLE_ACTUATOR_CFG,  # noqa: F401
                                          ANYDRIVE_3_LSTM_ACTUATOR_CFG,
                                          ANYMAL_B_CFG, ANYMAL_C_CFG, ANYMAL_D_CFG)

##
# Configuration - Actuators.
##

ANYDRIVE_3_EFFORT_ACTUATOR_CFG = EffortMotorCfg(
    joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
    effort_limit=80.0,
    velocity_limit=7.5,
    stiffness={".*": 100.0},
    damping={".*": 5.0},
)


ANYDRIVE_3_IMPLICIT_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
    effort_limit=80.0,
    velocity_limit=7.5,
    stiffness={".*": 100.0},
    damping={".*": 5.0},
)


"""Configuration for ANYdrive 3.0 (used on ANYmal-C) with LSTM actuator model."""


##
# Configuration - Articulation.
##
ANYMAL_B_CFG.replace(actuators={"legs": ANYDRIVE_3_LSTM_ACTUATOR_CFG})  # type: ignore
"""Configuration of ANYmal-B robot using actuator-net."""


ANYMAL_C_CFG = ANYMAL_C_CFG.replace(actuators={"legs": ANYDRIVE_3_LSTM_ACTUATOR_CFG})  # type: ignore
"""Configuration of ANYmal-C robot using actuator-net."""


ANYMAL_C_EFFORT_CFG = ANYMAL_C_CFG.replace(actuators={"legs": ANYDRIVE_3_EFFORT_ACTUATOR_CFG})
"""Configuration of ANYmal-C robot using actuator-net."""


ANYMAL_C_IMPLICIT_CFG = ANYMAL_C_CFG.replace(actuators={"legs": ANYDRIVE_3_IMPLICIT_ACTUATOR_CFG})
"""Configuration of ANYmal-C robot using actuator-net."""


ANYMAL_D_CFG = ANYMAL_D_CFG.replace(actuators={"legs": ANYDRIVE_3_LSTM_ACTUATOR_CFG})  # type: ignore
"""Configuration of ANYmal-D robot using actuator-net.

Note:
    Since we don't have a publicly available actuator network for ANYmal-D, we use the same network as ANYmal-C.
    This may impact the sim-to-real transfer performance.
"""


##
# Configuration - Sensors.
##

ANYMAL_LIDAR_CFG = VELODYNE_VLP_16_RAYCASTER_CFG.replace(  # type: ignore
    offset=RayCasterCfg.OffsetCfg(pos=(-0.310, 0.000, 0.159), rot=(0.0, 0.0, 0.0, 1.0))
)
"""Configuration for the Velodyne VLP-16 sensor mounted on the ANYmal robot's base."""
