# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents, fetching_env_cfg, fetching_terain_gen_cfg

##
# Register Gym environments.
##

gym.register(
    id="Octi-Fetching-Rough-Anymal-C-v0",
    entry_point="octi.lab.envs:OctiManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": fetching_env_cfg.AnymalCRoughPositionEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rough_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.AnymalCRoughFetchingPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)


gym.register(
    id="Octi-Fetching-Flat-Anymal-C-v0",
    entry_point="octi.lab.envs:OctiManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": fetching_env_cfg.AnymalCFlatPositionEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rough_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.AnymalCFlatFetchingPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

gym.register(
    id="Octi-Fetching-Generator-Anymal-C-v0",
    entry_point="octi.lab.envs:OctiManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": fetching_terain_gen_cfg.AnymalCGeneratorFetchingEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rough_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.AnymalCGeneratorFetchingPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)


