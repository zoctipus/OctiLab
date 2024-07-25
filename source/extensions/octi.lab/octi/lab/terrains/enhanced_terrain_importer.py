# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from omni.isaac.lab.terrains import TerrainImporter, TerrainImporterCfg


class EnhancedTerrainImporter(TerrainImporter):
    r"""A class to handle terrain meshes and import them into the simulator.

    We assume that a terrain mesh comprises of sub-terrains that are arranged in a grid with
    rows ``num_rows`` and columns ``num_cols``. The terrain origins are the positions of the sub-terrains
    where the robot should be spawned.

    Based on the configuration, the terrain importer handles computing the environment origins from the sub-terrain
    origins. In a typical setup, the number of sub-terrains (:math:`num\_rows \times num\_cols`) is smaller than
    the number of environments (:math:`num\_envs`). In this case, the environment origins are computed by
    sampling the sub-terrain origins.

    If a curriculum is used, it is possible to update the environment origins to terrain origins that correspond
    to a harder difficulty. This is done by calling :func:`update_terrain_levels`. The idea comes from game-based
    curriculum. For example, in a game, the player starts with easy levels and progresses to harder levels.
    """

    env_partitions: list[torch.Tensor]
    """The env partition for multi agent purpose. Shape is num_col x num_row entry, where each entry is Tensor of associated env_idx"""

    def __init__(self, cfg: TerrainImporterCfg):
        """Initialize the terrain importer.

        Args:
            cfg: The configuration for the terrain importer.

        Raises:
            ValueError: If input terrain type is not supported.
            ValueError: If terrain type is 'generator' and no configuration provided for ``terrain_generator``.
            ValueError: If terrain type is 'usd' and no configuration provided for ``usd_path``.
            ValueError: If terrain type is 'usd' or 'plane' and no configuration provided for ``env_spacing``.
        """
        super().__init__(cfg)

    def _compute_env_origins_curriculum(self, num_envs: int, origins: torch.Tensor) -> torch.Tensor:
        """Compute the origins of the environments defined by the sub-terrains origins."""
        env_origins = super()._compute_env_origins_curriculum(num_envs, origins)
        num_rows, num_cols = origins.shape[:2]
        self.terrain_levels = torch.div(
            torch.arange(num_envs, device=self.device),
            (num_envs / num_cols / num_rows),
            rounding_mode="floor"
        ).to(torch.long)
        self.env_partitions = [torch.nonzero(self.terrain_levels == i).squeeze() for i in range(num_cols * num_rows)]
        self.terrain_levels = self.terrain_levels % num_rows
        env_origins[:] = origins[self.terrain_levels, self.terrain_types]
        return env_origins
