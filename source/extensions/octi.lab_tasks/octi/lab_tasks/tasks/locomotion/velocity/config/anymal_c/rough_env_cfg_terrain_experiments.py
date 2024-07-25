# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
import omni.isaac.lab.terrains as terrain_gen
##
# Pre-defined configs
##
from .rough_env_cfg import AnymalCRoughEnvCfg
from omni.isaac.lab.sensors.camera import Camera, CameraCfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.managers import EventTermCfg as EventTerm
import octi.lab_tasks.tasks.locomotion.velocity.mdp as octi_mdp


@configclass
class AnymalCRoughEnvPrmdIprmdBoxRghHfslpCfg(AnymalCRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.terrain_generator.sub_terrains = {
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
                proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
            ),
            "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
            ),
            "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=0.2, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
            ),
            # "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            #     proportion=0.2, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
            # ),
        }


@configclass
class AnymalCRoughEnvPrmdIprmdBoxRghCfg(AnymalCRoughEnvPrmdIprmdBoxRghHfslpCfg):
    def __post_init__(self):
        # Call the parent post_init
        super().__post_init__()
        self.scene.terrain.terrain_generator.num_cols = 4
        # Remove the hf_pyramid_slope_inv key from the sub_terrains dictionary
        self.scene.terrain.terrain_generator.sub_terrains.pop("hf_pyramid_slope", None)


@configclass
class AnymalCRoughEnvPrmdIprmdBoxHfslpCfg(AnymalCRoughEnvPrmdIprmdBoxRghHfslpCfg):
    def __post_init__(self):
        # Call the parent post_init
        super().__post_init__()
        self.scene.terrain.terrain_generator.num_cols = 4
        # Remove the hf_pyramid_slope_inv key from the sub_terrains dictionary
        self.scene.terrain.terrain_generator.sub_terrains.pop("random_rough", None)


@configclass
class AnymalCRoughEnvPrmdIprmdRghHfslpCfg(AnymalCRoughEnvPrmdIprmdBoxRghHfslpCfg):
    def __post_init__(self):
        # Call the parent post_init
        super().__post_init__()
        self.scene.terrain.terrain_generator.num_cols = 4
        # Remove the hf_pyramid_slope_inv key from the sub_terrains dictionary
        self.scene.terrain.terrain_generator.sub_terrains.pop("boxes", None)


@configclass
class AnymalCRoughEnvPrmdBoxRghHfslpCfg(AnymalCRoughEnvPrmdIprmdBoxRghHfslpCfg):
    def __post_init__(self):
        # Call the parent post_init
        super().__post_init__()
        self.scene.terrain.terrain_generator.num_cols = 4
        # Remove the hf_pyramid_slope_inv key from the sub_terrains dictionary
        self.scene.terrain.terrain_generator.sub_terrains.pop("pyramid_stairs_inv", None)


@configclass
class AnymalCRoughEnvIprmdBoxRghHfslpCfg(AnymalCRoughEnvPrmdIprmdBoxRghHfslpCfg):
    def __post_init__(self):
        # Call the parent post_init
        super().__post_init__()
        self.scene.terrain.terrain_generator.num_cols = 4
        # Remove the hf_pyramid_slope_inv key from the sub_terrains dictionary
        self.scene.terrain.terrain_generator.sub_terrains.pop("pyramid_stairs", None)


@configclass
class AnymalCRoughEnvBoxRghHfslpCfg(AnymalCRoughEnvPrmdIprmdBoxRghHfslpCfg):
    def __post_init__(self):
        # Call the parent post_init
        super().__post_init__()
        self.scene.terrain.terrain_generator.num_cols = 3
        # Remove the hf_pyramid_slope_inv key from the sub_terrains dictionary
        self.scene.terrain.terrain_generator.sub_terrains.pop("pyramid_stairs", None)
        self.scene.terrain.terrain_generator.sub_terrains.pop("pyramid_stairs_inv", None)


@configclass
class AnymalCRoughEnvPrmdRghHfslpCfg(AnymalCRoughEnvPrmdIprmdBoxRghHfslpCfg):
    def __post_init__(self):
        # Call the parent post_init
        super().__post_init__()
        self.scene.terrain.terrain_generator.num_cols = 3
        # Remove the hf_pyramid_slope_inv key from the sub_terrains dictionary
        self.scene.terrain.terrain_generator.sub_terrains.pop("pyramid_stairs_inv", None)
        self.scene.terrain.terrain_generator.sub_terrains.pop("boxes", None)


@configclass
class AnymalCRoughEnvPrmdIprmdHfslpCfg(AnymalCRoughEnvPrmdIprmdBoxRghHfslpCfg):
    def __post_init__(self):
        # Call the parent post_init
        super().__post_init__()
        self.scene.terrain.terrain_generator.num_cols = 3
        # Remove the hf_pyramid_slope_inv key from the sub_terrains dictionary
        self.scene.terrain.terrain_generator.sub_terrains.pop("boxes", None)
        self.scene.terrain.terrain_generator.sub_terrains.pop("random_rough", None)


@configclass
class AnymalCRoughEnvPrmdIprmdBoxCfg(AnymalCRoughEnvPrmdIprmdBoxRghHfslpCfg):
    def __post_init__(self):
        # Call the parent post_init
        super().__post_init__()
        self.scene.terrain.terrain_generator.num_cols = 3
        # Remove the hf_pyramid_slope_inv key from the sub_terrains dictionary
        self.scene.terrain.terrain_generator.sub_terrains.pop("random_rough", None)
        self.scene.terrain.terrain_generator.sub_terrains.pop("hf_pyramid_slope", None)


@configclass
class AnymalCRoughEnvPrmdIprmdCfg(AnymalCRoughEnvPrmdIprmdBoxRghHfslpCfg):
    def __post_init__(self):
        # Call the parent post_init
        super().__post_init__()
        self.scene.terrain.terrain_generator.num_cols = 2
        # Remove the hf_pyramid_slope_inv key from the sub_terrains dictionary
        self.scene.terrain.terrain_generator.sub_terrains.pop("boxes", None)
        self.scene.terrain.terrain_generator.sub_terrains.pop("random_rough", None)
        self.scene.terrain.terrain_generator.sub_terrains.pop("hf_pyramid_slope", None)


@configclass
class AnymalCRoughEnvIprmdBoxCfg(AnymalCRoughEnvPrmdIprmdBoxRghHfslpCfg):
    def __post_init__(self):
        # Call the parent post_init
        super().__post_init__()
        self.scene.terrain.terrain_generator.num_cols = 2
        # Remove the hf_pyramid_slope_inv key from the sub_terrains dictionary
        self.scene.terrain.terrain_generator.sub_terrains.pop("pyramid_stairs", None)
        self.scene.terrain.terrain_generator.sub_terrains.pop("random_rough", None)
        self.scene.terrain.terrain_generator.sub_terrains.pop("hf_pyramid_slope", None)


@configclass
class AnymalCRoughEnvBoxRghCfg(AnymalCRoughEnvPrmdIprmdBoxRghHfslpCfg):
    def __post_init__(self):
        # Call the parent post_init
        super().__post_init__()
        self.scene.terrain.terrain_generator.num_cols = 2
        # Remove the hf_pyramid_slope_inv key from the sub_terrains dictionary
        self.scene.terrain.terrain_generator.sub_terrains.pop("pyramid_stairs", None)
        self.scene.terrain.terrain_generator.sub_terrains.pop("pyramid_stairs_inv", None)
        self.scene.terrain.terrain_generator.sub_terrains.pop("hf_pyramid_slope", None)


@configclass
class AnymalCRoughEnvRghHfslpCfg(AnymalCRoughEnvPrmdIprmdBoxRghHfslpCfg):
    def __post_init__(self):
        # Call the parent post_init
        super().__post_init__()
        self.scene.terrain.terrain_generator.num_cols = 2
        # Remove the hf_pyramid_slope_inv key from the sub_terrains dictionary
        self.scene.terrain.terrain_generator.sub_terrains.pop("pyramid_stairs", None)
        self.scene.terrain.terrain_generator.sub_terrains.pop("pyramid_stairs_inv", None)
        self.scene.terrain.terrain_generator.sub_terrains.pop("boxes", None)


@configclass
class AnymalCRoughEnvFlatCfg(AnymalCRoughEnvPrmdIprmdBoxRghHfslpCfg):
    def __post_init__(self):
        # Call the parent post_init
        super().__post_init__()
        self.scene.terrain.terrain_generator.num_cols = 1
        # Remove the hf_pyramid_slope_inv key from the sub_terrains dictionary
        self.scene.terrain.terrain_generator.sub_terrains = {
            "flat" : terrain_gen.MeshPlaneTerrainCfg(
                proportion=0.2,
            )
        }


@configclass
class AnymalCRoughEnvBoxCfg(AnymalCRoughEnvPrmdIprmdBoxRghHfslpCfg):
    def __post_init__(self):
        # Call the parent post_init
        super().__post_init__()
        self.scene.terrain.terrain_generator.num_cols = 10
        # Remove the hf_pyramid_slope_inv key from the sub_terrains dictionary
        self.scene.terrain.terrain_generator.sub_terrains = {
            "boxes": terrain_gen.MeshRandomGridTerrainCfg(
                proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
            ),
        }


@configclass
class AnymalCRoughEnvCameraCfg(AnymalCRoughEnvPrmdIprmdBoxRghHfslpCfg):
    def __post_init__(self):
        # Call the parent post_init
        super().__post_init__()
