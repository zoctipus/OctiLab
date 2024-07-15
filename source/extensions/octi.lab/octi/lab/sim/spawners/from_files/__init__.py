# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for spawners that spawn assets from files.

Currently, the following spawners are supported:

* :class:`UsdFileCfg`: Spawn an asset from a USD file.
* :class:`UrdfFileCfg`: Spawn an asset from a URDF file.
* :class:`GroundPlaneCfg`: Spawn a ground plane using the grid-world USD file.

"""

from .from_files import spawn_multi_object_randomly_sdf
from .from_files_cfg import MultiAssetCfg
