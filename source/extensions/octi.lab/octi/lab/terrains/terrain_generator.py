# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch
import trimesh

from omni.isaac.lab.utils.timer import Timer
from .terrain_generator_cfg import MultiOriginTerrainGeneratorCfg
from omni.isaac.lab.terrains.height_field import HfTerrainBaseCfg
from omni.isaac.lab.terrains.terrain_generator import TerrainGenerator
from omni.isaac.lab.terrains.utils import color_meshes_by_height


class MultiOriginTerrainGenerator(TerrainGenerator):
    
    def __init__(self, cfg: MultiOriginTerrainGeneratorCfg, device: str = "cpu") -> None:
        """Initialize the terrain generator.

        Args:
            cfg: Configuration for the terrain generator.
            device: The device to use for the flat patches tensor.
        """
        self.terrain_canditate_origins = [[torch.tensor([]) for _ in range(cfg.num_cols)] for _ in range(cfg.num_rows)]
        # check inputs
        if len(cfg.sub_terrains) == 0:
            raise ValueError("No sub-terrains specified! Please add at least one sub-terrain.")
        # store inputs
        self.cfg = cfg
        self.device = device
        # -- valid patches
        self.flat_patches = {}
        # set common values to all sub-terrains config
        for sub_cfg in self.cfg.sub_terrains.values():
            # size of all terrains
            sub_cfg.size = self.cfg.size
            # params for height field terrains
            if isinstance(sub_cfg, HfTerrainBaseCfg):
                sub_cfg.horizontal_scale = self.cfg.horizontal_scale
                sub_cfg.vertical_scale = self.cfg.vertical_scale
                sub_cfg.slope_threshold = self.cfg.slope_threshold

        # set the seed for reproducibility
        if self.cfg.seed is not None:
            torch.manual_seed(self.cfg.seed)
            np.random.seed(self.cfg.seed)
        # create a list of all sub-terrains
        self.terrain_meshes = list()
        self.terrain_origins = np.zeros((self.cfg.num_rows, self.cfg.num_cols, 3))

        # parse configuration and add sub-terrains
        # create terrains based on curriculum or randomly
        if self.cfg.curriculum:
            with Timer("[INFO] Generating terrains based on curriculum took"):
                self._generate_curriculum_terrains()
        else:
            terrain_proportions = [terrain.proportion for terrain in self.cfg.sub_terrains.values()]
            equal_proportion = all(x == terrain_proportions[0] for x in terrain_proportions)
            if equal_proportion:
                with Timer("[INFO] Generating terrains one by one took"):
                    self._generate_terrains_one_by_one()
            else:
                with Timer("[INFO] Generating terrains randomly took"):
                    self._generate_random_terrains()
        # add a border around the terrains
        self._add_terrain_border()
        # combine all the sub-terrains into a single mesh
        self.terrain_mesh = trimesh.util.concatenate(self.terrain_meshes)

        # color the terrain mesh
        if self.cfg.color_scheme == "height":
            self.terrain_mesh = color_meshes_by_height(self.terrain_mesh)
        elif self.cfg.color_scheme == "random":
            self.terrain_mesh.visual.vertex_colors = np.random.choice(
                range(256), size=(len(self.terrain_mesh.vertices), 4)
            )
        elif self.cfg.color_scheme == "none":
            pass
        else:
            raise ValueError(f"Invalid color scheme: {self.cfg.color_scheme}.")

        # offset the entire terrain and origins so that it is centered
        # -- terrain mesh
        transform = np.eye(4)
        transform[:2, -1] = -self.cfg.size[0] * self.cfg.num_rows * 0.5, -self.cfg.size[1] * self.cfg.num_cols * 0.5
        self.terrain_mesh.apply_transform(transform)
        # -- terrain origins
        self.terrain_origins += transform[:3, -1]
        # -- valid patches
        terrain_origins_torch = torch.tensor(self.terrain_origins, dtype=torch.float, device=self.device).unsqueeze(2)
        for name, value in self.flat_patches.items():
            self.flat_patches[name] = value + terrain_origins_torch
        
    def _generate_curriculum_terrains(self):
        """Add terrains based on the difficulty parameter."""
        # normalize the proportions of the sub-terrains
        proportions = np.array([sub_cfg.proportion for sub_cfg in self.cfg.sub_terrains.values()])
        proportions /= np.sum(proportions)

        # find the sub-terrain index for each column
        # we generate the terrains based on their proportion (not randomly sampled)
        sub_indices = []
        for index in range(self.cfg.num_cols):
            sub_index = np.min(np.where(index / self.cfg.num_cols + 0.001 < np.cumsum(proportions))[0])
            sub_indices.append(sub_index)
        sub_indices = np.array(sub_indices, dtype=np.int32)
        # create a list of all terrain configs
        sub_terrains_cfgs = list(self.cfg.sub_terrains.values())

        # curriculum-based sub-terrains
        for sub_col in range(self.cfg.num_cols):
            for sub_row in range(self.cfg.num_rows):
                # vary the difficulty parameter linearly over the number of rows
                lower, upper = self.cfg.difficulty_range
                difficulty = (sub_row + np.random.uniform()) / self.cfg.num_rows
                difficulty = lower + (upper - lower) * difficulty
                # generate terrain
                mesh, origin = self._get_terrain_mesh(difficulty, sub_terrains_cfgs[sub_indices[sub_col]])
                # resolve the origin on case whether terrain is multi origin or single origin
                origin = self.multi_origin_processing(origin, sub_row, sub_col)
                # add to sub-terrains
                self._add_sub_terrain(mesh, origin, sub_row, sub_col, sub_terrains_cfgs[sub_indices[sub_col]])
        
    def _generate_random_terrains(self):
        """Add terrains based on randomly sampled difficulty parameter."""
        # normalize the proportions of the sub-terrains
        proportions = np.array([sub_cfg.proportion for sub_cfg in self.cfg.sub_terrains.values()])
        proportions /= np.sum(proportions)
        # create a list of all terrain configs
        sub_terrains_cfgs = list(self.cfg.sub_terrains.values())

        # randomly sample sub-terrains
        for index in range(self.cfg.num_rows * self.cfg.num_cols):
            # coordinate index of the sub-terrain
            (sub_row, sub_col) = np.unravel_index(index, (self.cfg.num_rows, self.cfg.num_cols))
            # randomly sample terrain index
            sub_index = np.random.choice(len(proportions), p=proportions)
            # randomly sample difficulty parameter
            difficulty = np.random.uniform(*self.cfg.difficulty_range)
            # generate terrain
            mesh, origin = self._get_terrain_mesh(difficulty, sub_terrains_cfgs[sub_index])
            # resolve the origin on case whether terrain is multi origin or single origin
            origin = self.multi_origin_processing(origin, sub_row, sub_col)
            # add to sub-terrains
            self._add_sub_terrain(mesh, origin, sub_row, sub_col, sub_terrains_cfgs[sub_index])
    
    def _generate_terrains_one_by_one(self):
        proportions = np.array([sub_cfg.proportion for sub_cfg in self.cfg.sub_terrains.values()])
        proportions /= np.sum(proportions)
        # create a list of all terrain configs
        sub_terrains_cfgs = list(self.cfg.sub_terrains.values())
        terrain_added_count = 0
        # randomly sample sub-terrains
        for index in range(self.cfg.num_rows * self.cfg.num_cols):
            # coordinate index of the sub-terrain
            (sub_row, sub_col) = np.unravel_index(index, (self.cfg.num_rows, self.cfg.num_cols))
            # randomly sample terrain index
            sub_index = terrain_added_count % len(proportions)
            terrain_added_count += 1
            # randomly sample difficulty parameter
            difficulty = np.random.uniform(*self.cfg.difficulty_range)
            # generate terrain
            mesh, origin = self._get_terrain_mesh(difficulty, sub_terrains_cfgs[sub_index])
            # resolve the origin on case whether terrain is multi origin or single origin
            origin = self.multi_origin_processing(origin, sub_row, sub_col)
            # add to sub-terrains
            self._add_sub_terrain(mesh, origin, sub_row, sub_col, sub_terrains_cfgs[sub_index])
    
    def sample_origin(self, rows:torch.Tensor, cols:torch.Tensor):
        env_origins = torch.zeros((len(rows), 3), device=self.device)
        i = 0
        terrain_origins = torch.tensor(self.terrain_origins, device=self.device)
        for row, col in zip(rows, cols):
            terrain = self.terrain_canditate_origins[row][col]
            random_index = torch.randint(0, len(terrain), (1,))
            env_origins[i] =terrain_origins[row][col]  + terrain[random_index]
            i += 1
        return env_origins
    
    def multi_origin_processing(self, origin:np.ndarray, sub_row:int, sub_col:int):
        """
        multi origin terrain returns a origin of shape (n, 3) where origin[0] index represents the center of the terrain
        and origin[1:] represents available spawnable locations
        
        default terrain returns a origin of shape (1,3) and it represents the spawnable locations of the terrain
        
        this method reconcile both cases and treat both case as having both origin and spawnable location, 
        returning respective orgins store spawnable locations internally
        """
        if origin.ndim > 1:
            spawn_options = origin[1:]
            origin = origin[0]
        else:
            spawn_options = np.array([origin])
            origin[:2] = 0
        # sanity check that spawn_options is a (n,3) shape
        # sanity check that origin is (3,) shape
        assert(spawn_options.ndim == 2 and spawn_options.shape[1] == 3)
        assert(origin.ndim == 1 and origin.shape[0] == 3)
        # spawning_options already contains height information so origin height infomation needs to be 0 to avoid
        # double counting
        origin[2] = 0
        # transform the mesh to the correct position
        self.terrain_canditate_origins[sub_row][sub_col] = torch.tensor(spawn_options, device=self.device)
        return origin