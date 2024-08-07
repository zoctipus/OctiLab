# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from omni.isaac.lab.terrains import TerrainImporter, TerrainImporterCfg
from .terrain_generator import MultiOriginTerrainGenerator
import omni.isaac.lab.sim as sim_utils

class MultiAgentTerrainImporter(TerrainImporter):

    env_partitions: list[torch.Tensor]
    """The env partition for multi agent purpose. Shape is num_col x num_row entry, where each entry is Tensor of associated env_idx"""

    def __init__(self, cfg: TerrainImporterCfg):
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

class MultiOriginTerrainImporter(TerrainImporter):
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
        # store inputs
        self.cfg = cfg
        self.device = sim_utils.SimulationContext.instance().device  # type: ignore

        # create a dict of meshes
        self.meshes = dict()
        self.warp_meshes = dict()
        self.env_origins = None
        self.terrain_origins = None
        # private variables
        self._terrain_flat_patches = dict()
        self.terrain_generator = None

        # auto-import the terrain based on the config
        if self.cfg.terrain_type == "generator":
            # check config is provided
            if self.cfg.terrain_generator is None:
                raise ValueError("Input terrain type is 'generator' but no value provided for 'terrain_generator'.")
            # generate the terrain
            self.terrain_generator = MultiOriginTerrainGenerator(cfg=self.cfg.terrain_generator, device=self.device)
            self.import_mesh("terrain", self.terrain_generator.terrain_mesh)
            # configure the terrain origins based on the terrain generator
            self.configure_env_origins(self.terrain_generator.terrain_origins)
            # refer to the flat patches
            self._terrain_flat_patches = self.terrain_generator.flat_patches
        elif self.cfg.terrain_type == "usd":
            # check if config is provided
            if self.cfg.usd_path is None:
                raise ValueError("Input terrain type is 'usd' but no value provided for 'usd_path'.")
            # import the terrain
            self.import_usd("terrain", self.cfg.usd_path)
            # configure the origins in a grid
            self.configure_env_origins()
        elif self.cfg.terrain_type == "plane":
            # load the plane
            self.import_ground_plane("terrain")
            # configure the origins in a grid
            self.configure_env_origins()
        else:
            raise ValueError(f"Terrain type '{self.cfg.terrain_type}' not available.")

        # set initial state of debug visualization
        self.set_debug_vis(self.cfg.debug_vis)
        
    def update_env_origins(self, env_ids: torch.Tensor, move_up: torch.Tensor, move_down: torch.Tensor):
        """Update the environment origins based on the terrain levels."""
        # check if grid-like spawning
        if self.terrain_origins is None:
            return
        # update terrain level for the envs
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # robots that solve the last level are sent to a random one
        # the minimum level is zero
        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
            torch.clip(self.terrain_levels[env_ids], 0),
        )
        # update the env origins
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    def reset_env_origins(self, env_ids: torch.Tensor):
        if self.terrain_origins is None:
            return
        new_origin = self.terrain_generator.sample_origin(self.terrain_levels[env_ids], self.terrain_types[env_ids])
        # update the env origins
        self.env_origins[env_ids] = new_origin
