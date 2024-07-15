# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING
from omni.isaac.lab.sim.spawners.spawner_cfg import SpawnerCfg
from omni.isaac.lab.utils import configclass

from . import from_files


@configclass
class MultiAssetCfg(SpawnerCfg):
    """Configuration parameters for loading multiple assets randomly."""

    # Uncomment this one: 45 seconds for 2048 envs
    # func: Callable = from_files.spawn_multi_object_randomly
    # Uncomment this one: 2.15 seconds for 2048 envs
    func: Callable = from_files.spawn_multi_object_randomly_sdf

    assets_cfg: list[SpawnerCfg] = MISSING  # type: ignore
    """List of asset configurations to spawn."""
