import gymnasium as gym
from . import agents
from . import track_goal_leap_xarm, track_goal_leap

"""
LeapXarm
"""
gym.register(
    id="Octi-Track-Goal-LeapXarm-JointPos-v0",
    entry_point="octi.lab.envs.octi_manager_based_rl:OctiManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": track_goal_leap_xarm.TrackGoalLeapXarmJointPosition,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.TrackGoalLeapXarmPPORunnerCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Octi-Track-Goal-LeapXarm-McIkAbs-v0",
    entry_point="octi.lab.envs.octi_manager_based_rl:OctiManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": track_goal_leap_xarm.TrackGoalLeapXarmMcIkAbs,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.TrackGoalLeapXarmPPORunnerCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
    },
    disable_env_checker=True,
)


gym.register(
    id="Octi-Track-Goal-LeapXarm-McIkDel-v0",
    entry_point="octi.lab.envs.octi_manager_based_rl:OctiManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": track_goal_leap_xarm.TrackGoalLeapXarmMcIkDel,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.TrackGoalLeapXarmPPORunnerCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
    },
    disable_env_checker=True,
)

"""
Leap
"""
gym.register(
    id="Octi-Track-Goal-Leap-JointPos-v0",
    entry_point="octi.lab.envs.octi_manager_based_rl:OctiManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": track_goal_leap.TrackGoalLeapJointPosition,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.TrackGoalLeapPPORunnerCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
    },
    disable_env_checker=True,
)


gym.register(
    id="Octi-Track-Goal-Leap-McIkAbs-v0",
    entry_point="octi.lab.envs.octi_manager_based_rl:OctiManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": track_goal_leap.TrackGoalLeapMcIkAbs,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.TrackGoalLeapPPORunnerCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
    },
    disable_env_checker=True,
)


gym.register(
    id="Octi-Track-Goal-Leap-McIkDel-v0",
    entry_point="octi.lab.envs.octi_manager_based_rl:OctiManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": track_goal_leap.TrackGoalLeapMcIkDel,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.TrackGoalLeapPPORunnerCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
    },
    disable_env_checker=True,
)