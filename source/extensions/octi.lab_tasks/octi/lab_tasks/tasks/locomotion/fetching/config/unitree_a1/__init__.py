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
    id="Octi-Fetching-Rough-Unitree-A1-v0",
    entry_point="octi.lab.envs:OctiManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.UnitreeA1FetchingRoughCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rough_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.UnitreeA1RoughPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)


gym.register(
    id="Octi-Fetching-Flat-Unitree-A1-v0",
    entry_point="octi.lab.envs:OctiManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.UnitreeA1FetchingFlatCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rough_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.UnitreeA1RoughPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)
