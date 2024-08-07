# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import omni.isaac.lab.terrains as terrain_gen
import octi.lab.terrains as octi_terrain_gen

from omni.isaac.lab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from octi.lab.terrains.terrain_generator_cfg import MultiOriginTerrainGeneratorCfg

TERRAINS_GEN_CFG = MultiOriginTerrainGeneratorCfg(
    size=(40.0, 40.0),
    border_width=20.0,
    num_rows=2,
    num_cols=2,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "obj_terrain0": octi_terrain_gen.MeshObjTerrainCfg(
            obj_path='/home/octipus/Projects/terrain-generator/results/height2.5/mesh_0/mesh_terrain.obj',
            spawn_origin_path='/home/octipus/Projects/terrain-generator/results/height2.5/mesh_0/spawnable_locations.npy'
        ),
        "obj_terrain1": octi_terrain_gen.MeshObjTerrainCfg(
            obj_path='/home/octipus/Projects/terrain-generator/results/height2.5/mesh_1/mesh_terrain.obj',
            spawn_origin_path='/home/octipus/Projects/terrain-generator/results/height2.5/mesh_1/spawnable_locations.npy'
        ),
        "obj_terrain2": octi_terrain_gen.MeshObjTerrainCfg(
            obj_path='/home/octipus/Projects/terrain-generator/results/height2.5/mesh_2/mesh_terrain.obj',
            spawn_origin_path='/home/octipus/Projects/terrain-generator/results/height2.5/mesh_2/spawnable_locations.npy'
        ),
        "obj_terrain3": octi_terrain_gen.MeshObjTerrainCfg(
            obj_path='/home/octipus/Projects/terrain-generator/results/height2.5/mesh_3/mesh_terrain.obj',
            spawn_origin_path='/home/octipus/Projects/terrain-generator/results/height2.5/mesh_3/spawnable_locations.npy'
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
)

TERRAINS_GEN_PLAYGROUND_CFG = MultiOriginTerrainGeneratorCfg(
    size=(16.0, 16.0),
    border_width=20.0,
    num_rows=4,
    num_cols=4,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "perlin": octi_terrain_gen.MeshObjTerrainCfg(
            proportion=0.08,
            obj_path='datasets/terrains/perlin/mesh_terrain.obj',
            spawn_origin_path='datasets/terrains/perlin/spawnable_locations.npy'
        ),
        "perlin_ramp": octi_terrain_gen.MeshObjTerrainCfg(
            proportion=0.08,
            obj_path='datasets/terrains/perlin_ramp/mesh_terrain.obj',
            spawn_origin_path='datasets/terrains/perlin_ramp/spawnable_locations.npy'
        ),
        "wall": octi_terrain_gen.MeshObjTerrainCfg(
            proportion=0.08,
            obj_path='datasets/terrains/wall/mesh_terrain.obj',
            spawn_origin_path='datasets/terrains/wall/spawnable_locations.npy'
        ),
        "stair_ramp": octi_terrain_gen.MeshObjTerrainCfg(
            proportion=0.08,
            obj_path='datasets/terrains/stair_ramp/mesh_terrain.obj',
            spawn_origin_path='datasets/terrains/stair_ramp/spawnable_locations.npy'
        ),
        "stair_platform": octi_terrain_gen.MeshObjTerrainCfg(
            proportion=0.08,
            obj_path='datasets/terrains/stair_platform/mesh_terrain.obj',
            spawn_origin_path='datasets/terrains/stair_platform/spawnable_locations.npy'
        ),
        "ramp": octi_terrain_gen.MeshObjTerrainCfg(
            proportion=0.08,
            obj_path='datasets/terrains/ramp/mesh_terrain.obj',
            spawn_origin_path='datasets/terrains/ramp/spawnable_locations.npy'
        ),
        "wall_stair_platform": octi_terrain_gen.MeshObjTerrainCfg(
            proportion=0.08,
            obj_path='datasets/terrains/wall_stair_platform/mesh_terrain.obj',
            spawn_origin_path='datasets/terrains/wall_stair_platform/spawnable_locations.npy'
        ),
        "stair_platform2": octi_terrain_gen.MeshObjTerrainCfg(
            proportion=0.08,
            obj_path='datasets/terrains/stair_platform2/mesh_terrain.obj',
            spawn_origin_path='datasets/terrains/stair_platform2/spawnable_locations.npy'
        ),
        "perlin_wall": octi_terrain_gen.MeshObjTerrainCfg(
            proportion=0.08,
            obj_path='datasets/terrains/perlin_wall/mesh_terrain.obj',
            spawn_origin_path='datasets/terrains/perlin_wall/spawnable_locations.npy'
        ),
        "box": octi_terrain_gen.MeshObjTerrainCfg(
            proportion=0.08,
            obj_path='datasets/terrains/box/mesh_terrain.obj',
            spawn_origin_path='datasets/terrains/box/spawnable_locations.npy'
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.08,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.08,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.08, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.08, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.08, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "random_grid": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.08,
            platform_width=1.5,
            grid_width=0.75,
            grid_height_range=(0.025, 0.2),
            holes=False,
        )
    },
)
"""Rough terrains configuration."""


ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
)
"""Rough terrains configuration."""
