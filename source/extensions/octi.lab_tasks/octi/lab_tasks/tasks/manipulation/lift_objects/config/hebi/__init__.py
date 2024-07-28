import gymnasium as gym
from . import agents
from . import lift_cube_tycho

gym.register(
    id="Octi-LiftCube-Tycho-IkDel-v0",
    entry_point="octi.lab.envs.octi_manager_based_rl:OctiManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": lift_cube_tycho.LiftCubeTychoIkdelta,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Base_PPORunnerCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
        "d3rlpy_cfg_entry_point": f"{agents.__name__}:d3rlpy_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Octi-LiftCube-Tycho-IkAbs-v0",
    entry_point="octi.lab.envs.octi_manager_based_rl:OctiManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": lift_cube_tycho.LiftCubeTychoIkabsolute,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Base_PPORunnerCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
        "d3rlpy_cfg_entry_point": f"{agents.__name__}:d3rlpy_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Octi-LiftCube-Tycho-JointPos-v0",
    entry_point="octi.lab.envs.octi_manager_based_rl:OctiManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": lift_cube_tycho.LiftCubeTychoJointPosition,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Base_PPORunnerCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
        "d3rlpy_cfg_entry_point": f"{agents.__name__}:d3rlpy_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Octi-LiftCube-Tycho-JointEff-v0",
    entry_point="octi.lab.envs.octi_manager_based_rl:OctiManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": lift_cube_tycho.LiftCubeTychoJointEffort,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Base_PPORunnerCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
        "d3rlpy_cfg_entry_point": f"{agents.__name__}:d3rlpy_cfg.yaml",
    },
    disable_env_checker=True,
)
