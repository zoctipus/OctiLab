import gymnasium as gym
from . import agents
from . import lift_objects_leap_xarm

gym.register(
    id="Octi-LiftObjects-LeapXarm-JointPos-v0",
    entry_point="octi.lab.envs.octi_manager_based_rl:OctiManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": lift_objects_leap_xarm.LiftObejctsLeapXarmJointPosition,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Base_PPORunnerCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Octi-LiftObjects-LeapXarm-IkAbs-v0",
    entry_point="octi.lab.envs.octi_manager_based_rl:OctiManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": lift_objects_leap_xarm.LiftObejctsLeapXarmMcIkAbs,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Base_PPORunnerCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
    },
    disable_env_checker=True,
)


gym.register(
    id="Octi-LiftObjects-LeapXarm-IkDel-v0",
    entry_point="octi.lab.envs.octi_manager_based_rl:OctiManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": lift_objects_leap_xarm.LiftObejctsLeapXarmMcIkDel,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Base_PPORunnerCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
    },
    disable_env_checker=True,
)