from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from omni.isaac.lab.managers import SceneEntityCfg
from octi.lab.terrains import MultiOriginTerrainImporter

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv


def reset_origin(env: ManagerBasedEnv, env_ids: torch.Tensor, multi_origin_terrain: SceneEntityCfg):
    """Reset the scene to the default state specified in the scene configuration."""
    terrain: MultiOriginTerrainImporter = env.scene[multi_origin_terrain.name]
    terrain.reset_env_origins(env_ids)