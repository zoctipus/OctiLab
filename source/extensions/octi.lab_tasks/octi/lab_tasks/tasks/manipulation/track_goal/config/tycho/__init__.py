import gymnasium as gym
from . import agents
from . import tycho_track_goal

gym.register(
    id="Octi-Track-Goal-Tycho-IkDel-v0",
    entry_point="octi.lab.envs.octi_manager_based_rl:OctiManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": tycho_track_goal.GoalTrackingTychoIkdelta,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Base_PPORunnerCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
        "d3rlpy_cfg_entry_point": f"{agents.__name__}:d3rlpy_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Octi-Track-Goal-Tycho-IkAbs-v0",
    entry_point="octi.lab.envs.octi_manager_based_rl:OctiManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": tycho_track_goal.GoalTrackingTychoIkabsolute,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Base_PPORunnerCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
        "d3rlpy_cfg_entry_point": f"{agents.__name__}:d3rlpy_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Octi-Track-Goal-Tycho-JointPos-v0",
    entry_point="octi.lab.envs.octi_manager_based_rl:OctiManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": tycho_track_goal.GoalTrackingTychoJointPosition,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Base_PPORunnerCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
        "d3rlpy_cfg_entry_point": f"{agents.__name__}:d3rlpy_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Octi-Track-Goal-Tycho-JointEff-v0",
    entry_point="octi.lab.envs.octi_manager_based_rl:OctiManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": tycho_track_goal.GoalTrackingTychoJointEffort,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Base_PPORunnerCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
        "d3rlpy_cfg_entry_point": f"{agents.__name__}:d3rlpy_cfg.yaml",
    },
    disable_env_checker=True,
)