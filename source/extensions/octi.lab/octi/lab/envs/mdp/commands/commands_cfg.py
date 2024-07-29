# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from omni.isaac.lab.managers import CommandTermCfg
from omni.isaac.lab.utils import configclass
from .categorical_command import CategoricalCommand


@configclass
class CategoricalCommandCfg(CommandTermCfg):
    """Configuration for the uniform velocity command generator."""

    class_type: type = CategoricalCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    
    num_category: int = MISSING
