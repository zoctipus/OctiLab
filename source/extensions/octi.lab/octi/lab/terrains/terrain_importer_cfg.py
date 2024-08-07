# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils import configclass

from .terrain_importer import MultiOriginTerrainImporter
from omni.isaac.lab.terrains import TerrainImporterCfg
if TYPE_CHECKING:
    from .terrain_generator_cfg import TerrainGeneratorCfg


@configclass
class MultiOriginTerrainImporterCfg(TerrainImporterCfg):
    """Configuration for the terrain manager."""

    class_type: type = MultiOriginTerrainImporter
    