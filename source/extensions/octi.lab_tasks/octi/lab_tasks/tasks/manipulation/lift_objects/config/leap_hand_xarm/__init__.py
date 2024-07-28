import gymnasium as gym
from . import agents
from . import leap_xarm_lift_objects

gym.register(
    id="Octi-Lift-Objects-LeapXarm-JointPos-v0",
    entry_point="octi.lab.envs.octi_manager_based_rl:OctiManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": leap_xarm_lift_objects.ObejctsLiftLeapXarmJointPosition,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Base_PPORunnerCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Octi-Lift-Objects-LeapXarm-McIkAbs-v0",
    entry_point="octi.lab.envs.octi_manager_based_rl:OctiManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": leap_xarm_lift_objects.ObejctsLiftLeapXarmMcIkAbs,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Base_PPORunnerCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
    },
    disable_env_checker=True,
)