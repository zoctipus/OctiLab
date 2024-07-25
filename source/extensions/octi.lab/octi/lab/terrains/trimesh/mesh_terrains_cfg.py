# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import octi.lab.terrains.trimesh.mesh_terrains as mesh_terrains
from omni.isaac.lab.utils import configclass

from omni.isaac.lab.terrains.terrain_generator_cfg import SubTerrainBaseCfg

"""
Different trimesh terrain configurations.
"""

@configclass
class MeshObjTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a plane mesh terrain."""

    function = mesh_terrains.obj_terrain

    obj_path : str = MISSING