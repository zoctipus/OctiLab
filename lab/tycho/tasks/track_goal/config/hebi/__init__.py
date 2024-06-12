import gymnasium as gym
from octilab.envs.create_env import create_hebi_env
import lab.tycho.cfgs.robots.hebi.robot_dynamics as rd
from .Hebi_JointPos_GoalTracking_Env import (ImplicitMotorOriginHebi_JointPos_GoalTracking_Env,
                                             PwmMotorHebi_JointPos_GoalTracking_Env,
                                             IdealPDHebi_JointPos_GoalTracking_Env,
                                             ImplicitMotorHebi_JointPos_GoalTracking_Env)
from . import agents

base_envs = [PwmMotorHebi_JointPos_GoalTracking_Env, 
             IdealPDHebi_JointPos_GoalTracking_Env, 
             ImplicitMotorHebi_JointPos_GoalTracking_Env]

action_classes = [rd.RobotActionsCfg_HebiIkDeltaDls, 
                rd.RobotActionsCfg_HebiIkAbsoluteDls, 
                rd.RobotActionsCfg_HebiCustomIkAbsolute, 
                rd.RobotActionsCfg_HebiCustomIkDelta]


# Loop through each configuration and register the environment
for base_env in base_envs:
    for action_class in action_classes:
        action_class_id = action_class.__name__.replace("RobotActionsCfg_Hebi", "")
        base_env_id = base_env.__name__.replace("_Env", "")
        _id = f"{action_class_id}_{base_env_id}_v0".replace("_", "-")
        gym.register(
            id=_id,
            entry_point="octilab.envs.hebi_rl_task_env:HebiRLTaskEnv",
            kwargs={
                "env_cfg_entry_point": create_hebi_env(
                                            base_env_cfg=base_env,
                                            rd_action_class=action_class
                                        ),
                "rsl_rl_cfg_entry_point": agents.rsl_rl_hebi_agent_cfg.Base_PPORunnerCfg,
                "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
                "d3rlpy_cfg_entry_point": f"{agents.__name__}:d3rlpy_cfg.yaml",
            },
            disable_env_checker=True,
        )

