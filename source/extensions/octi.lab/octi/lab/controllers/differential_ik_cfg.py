# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from omni.isaac.lab.utils import configclass

from .differential_ik import MultiConstraintDifferentialIKController
from omni.isaac.lab.controllers.differential_ik_cfg import DifferentialIKControllerCfg


@configclass
class MultiConstraintDifferentialIKControllerCfg(DifferentialIKControllerCfg):
    """Configuration for differential inverse kinematics controller."""

    class_type: type = MultiConstraintDifferentialIKController
