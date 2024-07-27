# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents, flat_env_cfg, rough_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Octi-Velocity-Flat-Anymal-C-v0",
    entry_point="octi.lab.envs:OctiManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.AnymalCRoughEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.AnymalCRoughPPORunnerFlatCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="Octi-Velocity-Flat-Anymal-C-Play-v0",
    entry_point="octi.lab.envs:OctiManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.AnymalCFlatEnvCfg_PLAY,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.AnymalCRoughPPORunnerFlatCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)



gym.register(
    id="Octi-Velocity-Rough-Anymal-C-v0",
    entry_point="octi.lab.envs:OctiManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.AnymalCRoughEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rough_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.AnymalCRoughPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

gym.register(
    id="Octi-Velocity-Rough-Anymal-C-Play-v0",
    entry_point="octi.lab.envs:OctiManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.AnymalCRoughEnvCfg_PLAY,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rough_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.AnymalCRoughPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

from . import rough_env_cfg_terrain_experiments
gym.register(
    id="Octi-Velocity-Rough-Anymal-C-Prmd-Iprmd-Box-Rgh-Hfslp-v0",
    entry_point="octi.lab.envs:OctiManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg_terrain_experiments.AnymalCRoughEnvPrmdIprmdBoxRghHfslpCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rough_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.AnymalCRoughPPORunnerPrmdIprmdBoxRghHfslpCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)
 
gym.register(
    id="Octi-Velocity-Rough-Anymal-C-Prmd-Iprmd-Box-Rgh-v0",
    entry_point="octi.lab.envs:OctiManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg_terrain_experiments.AnymalCRoughEnvPrmdIprmdBoxRghCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rough_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.AnymalCRoughPPORunnerPrmdIprmdBoxRghHfslpCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

gym.register(
    id="Octi-Velocity-Rough-Anymal-C-Prmd-Iprmd-Box-Hfslp-v0",
    entry_point="octi.lab.envs:OctiManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg_terrain_experiments.AnymalCRoughEnvPrmdIprmdBoxHfslpCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rough_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.AnymalCRoughPPORunnerPrmdIprmdBoxRghHfslpCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

gym.register(
    id="Octi-Velocity-Rough-Anymal-C-Prmd-Iprmd-Rgh-Hfslp-v0",
    entry_point="octi.lab.envs:OctiManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg_terrain_experiments.AnymalCRoughEnvPrmdIprmdRghHfslpCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rough_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.AnymalCRoughPPORunnerPrmdIprmdRghHfslpCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

gym.register(
    id="Octi-Velocity-Rough-Anymal-C-Prmd-Box-Rgh-Hfslp-v0",
    entry_point="octi.lab.envs:OctiManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg_terrain_experiments.AnymalCRoughEnvPrmdBoxRghHfslpCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rough_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.AnymalCRoughPPORunnerPrmdBoxRghHfslpCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

gym.register(
    id="Octi-Velocity-Rough-Anymal-C-Iprmd-Box-Rgh-Hfslp-v0",
    entry_point="octi.lab.envs:OctiManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg_terrain_experiments.AnymalCRoughEnvIprmdBoxRghHfslpCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rough_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.AnymalCRoughPPORunnerIprmdBoxRghHfslpCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

gym.register(
    id="Octi-Velocity-Rough-Anymal-C-Box-Rgh-Hfslp-v0",
    entry_point="octi.lab.envs:OctiManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg_terrain_experiments.AnymalCRoughEnvBoxRghHfslpCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rough_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.AnymalCRoughPPORunnerBoxRghHfslpCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

gym.register(
    id="Octi-Velocity-Rough-Anymal-C-Prmd-Rgh-Hfslp-v0",
    entry_point="octi.lab.envs:OctiManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg_terrain_experiments.AnymalCRoughEnvPrmdRghHfslpCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rough_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.AnymalCRoughPPORunnerPrmdRghHfslpCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

gym.register(
    id="Octi-Velocity-Rough-Anymal-C-Prmd-Iprmd-Hfslp-v0",
    entry_point="octi.lab.envs:OctiManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg_terrain_experiments.AnymalCRoughEnvPrmdIprmdHfslpCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rough_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.AnymalCRoughPPORunnerPrmdIprmdHfslpCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

gym.register(
    id="Octi-Velocity-Rough-Anymal-C-Prmd-Iprmd-v0",
    entry_point="octi.lab.envs:OctiManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg_terrain_experiments.AnymalCRoughEnvPrmdIprmdCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rough_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.AnymalCRoughPPORunnerPrmdIprmdCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

gym.register(
    id="Octi-Velocity-Rough-Anymal-C-Iprmd-Box-v0",
    entry_point="octi.lab.envs:OctiManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg_terrain_experiments.AnymalCRoughEnvIprmdBoxCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rough_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.AnymalCRoughPPORunnerIprmdBoxCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

gym.register(
    id="Octi-Velocity-Rough-Anymal-C-Box-Rgh-v0",
    entry_point="octi.lab.envs:OctiManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg_terrain_experiments.AnymalCRoughEnvBoxRghCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rough_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.AnymalCRoughPPORunnerBoxRghCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

gym.register(
    id="Octi-Velocity-Rough-Anymal-C-Rgh-Hfslp-v0",
    entry_point="octi.lab.envs:OctiManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg_terrain_experiments.AnymalCRoughEnvRghHfslpCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rough_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.AnymalCRoughPPORunnerRghHfslpCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

gym.register(
    id="Octi-Velocity-Rough-Anymal-C-Flat-v0",
    entry_point="octi.lab.envs:OctiManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg_terrain_experiments.AnymalCRoughEnvFlatCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rough_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.AnymalCRoughPPORunnerFlatCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

gym.register(
    id="Octi-Velocity-Rough-Anymal-C-Box-v0",
    entry_point="octi.lab.envs:OctiManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg_terrain_experiments.AnymalCRoughEnvBoxCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rough_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.AnymalCRoughPPORunnerBoxCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

gym.register(
    id="Octi-Velocity-Rough-Anymal-C-Camera-v0",
    entry_point="octi.lab.envs:OctiManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg_terrain_experiments.AnymalCRoughEnvCameraCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rough_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.AnymalCRoughPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)