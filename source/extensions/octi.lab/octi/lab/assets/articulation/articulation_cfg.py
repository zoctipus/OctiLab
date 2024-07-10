# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass
from omni.isaac.lab.assets.articulation.articulation_cfg import ArticulationCfg
from .articulation import HebiArticulation


@configclass
class HebiArticulationCfg(ArticulationCfg):
    """Configuration parameters for an articulation."""

    class_type: type = HebiArticulation
