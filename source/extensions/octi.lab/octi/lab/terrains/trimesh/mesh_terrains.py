# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions to generate different terrains using the ``trimesh`` library."""

from __future__ import annotations

import numpy as np
import scipy.spatial.transform as tf
import torch
import trimesh
from typing import TYPE_CHECKING

from .utils import *  # noqa: F401, F403
from .utils import make_border, make_plane

if TYPE_CHECKING:
    from . import mesh_terrains_cfg


def obj_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshObjTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    mesh: trimesh.Geometry = trimesh.load(cfg.obj_path)
    scale = cfg.size / (mesh.bounds[1] - mesh.bounds[0])[:2]
    # set the height scale to the average between length and width scale to perserve as much original shap as possible
    height_scale = (scale[0] + scale[1]) / 2
    mesh.apply_scale(np.array([*scale, height_scale]))
    mesh.apply_translation(-mesh.bounds[0])
    extend = mesh.bounds[1] - mesh.bounds[0]
    origin = (*((extend[:2]) / 2), mesh.bounds[1][2] / 2)
    return [mesh], np.array(origin)
