import gymnasium as gym
from octilab.envs.create_env import create_hebi_env
import ext.envs.cfgs.robots.hebi.robot_dynamics as rd
from . import agents
from .Hebi_JointPos_GoalTracking_Env import (
    EffortHebi_JointPos_GoalTracking_Env,
    Strategy3MotorHebi_JointPos_GoalTracking_Env,
    IdealPDHebi_JointPos_GoalTracking_Env,
    ImplicitMotorHebi_JointPos_GoalTracking_Env,
    Strategy4MotorHebi_JointPos_GoalTracking_Env,
    HebiDCMotorHebi_JointPos_GoalTracking_Env,
    DCMotorHebi_JointPos_GoalTracking_Env,
)

from .decimation_experiments import (
    IdealPDHebi_JointPos_GoalTracking_Decimate5_Env,
    IdealPDHebi_JointPos_GoalTracking_Decimate2_Env,
    IdealPDHebi_JointPos_GoalTracking_Decimate1_Env)

from .agent_update_rate_experiments import (
    IdealPDHebi_JointPos_GoalTracking_Agent4Hz_Env,
    IdealPDHebi_JointPos_GoalTracking_Agent6dot25Hz_Env,
    IdealPDHebi_JointPos_GoalTracking_Agent11dot11Hz_Env,
    IdealPDHebi_JointPos_GoalTracking_Agent25Hz_Env,
    IdealPDHebi_JointPos_GoalTracking_Agent100Hz_Env)

from .strategy3_scale_experiments import (
    Strategy3MotorHebi_JointPos_GoalTracking_pp0dot5_ep1_Env,
    Strategy3MotorHebi_JointPos_GoalTracking_pp1dot5_ep1_Env,
    Strategy3MotorHebi_JointPos_GoalTracking_pp2_ep1_Env,
    Strategy3MotorHebi_JointPos_GoalTracking_pp5_ep1_Env,
    Strategy3MotorHebi_JointPos_GoalTracking_pp1_ep0dot2_Env,
    Strategy3MotorHebi_JointPos_GoalTracking_pp1_ep0dot5_Env,
    Strategy3MotorHebi_JointPos_GoalTracking_pp1_ep1dot5_Env,
    Strategy3MotorHebi_JointPos_GoalTracking_pp1_ep2_Env,
)

base_envs = [
    EffortHebi_JointPos_GoalTracking_Env,
    Strategy3MotorHebi_JointPos_GoalTracking_Env,
    IdealPDHebi_JointPos_GoalTracking_Env,
    ImplicitMotorHebi_JointPos_GoalTracking_Env,
    Strategy4MotorHebi_JointPos_GoalTracking_Env,
    HebiDCMotorHebi_JointPos_GoalTracking_Env,
    DCMotorHebi_JointPos_GoalTracking_Env,
]

decimation_experiments_envs = [
    IdealPDHebi_JointPos_GoalTracking_Decimate5_Env,
    IdealPDHebi_JointPos_GoalTracking_Decimate2_Env,
    IdealPDHebi_JointPos_GoalTracking_Decimate1_Env,
]

agent_update_rate_experiments_envs = [
    IdealPDHebi_JointPos_GoalTracking_Agent4Hz_Env,
    IdealPDHebi_JointPos_GoalTracking_Agent6dot25Hz_Env,
    IdealPDHebi_JointPos_GoalTracking_Agent11dot11Hz_Env,
    IdealPDHebi_JointPos_GoalTracking_Agent25Hz_Env,
    IdealPDHebi_JointPos_GoalTracking_Agent100Hz_Env,
]

strategy3_experiment_envs = [
    Strategy3MotorHebi_JointPos_GoalTracking_pp0dot5_ep1_Env,
    Strategy3MotorHebi_JointPos_GoalTracking_pp1dot5_ep1_Env,
    Strategy3MotorHebi_JointPos_GoalTracking_pp2_ep1_Env,
    Strategy3MotorHebi_JointPos_GoalTracking_pp5_ep1_Env,
    Strategy3MotorHebi_JointPos_GoalTracking_pp1_ep0dot2_Env,
    Strategy3MotorHebi_JointPos_GoalTracking_pp1_ep0dot5_Env,
    Strategy3MotorHebi_JointPos_GoalTracking_pp1_ep1dot5_Env,
    Strategy3MotorHebi_JointPos_GoalTracking_pp1_ep2_Env,
]

action_classes = [
    rd.RobotActionsCfg_HebiIkDeltaDls,
    rd.RobotActionsCfg_HebiIkAbsoluteDls,
    rd.RobotActionsCfg_HebiJointPosition,
    rd.RobotActionsCfg_HebiJointEffort,
]


# Loop through each configuration and register the environment
for base_env in base_envs:
    for action_class in action_classes:
        action_class_id = action_class.__name__.replace("RobotActionsCfg_Hebi", "")
        base_env_id = base_env.__name__.replace("_Env", "")
        _id = f"{action_class_id}_{base_env_id}".replace("_", "-")
        gym.register(
            id=_id,
            entry_point="octilab.envs.octi_manager_based_rl:OctiManagerBasedRLEnv",
            kwargs={
                "env_cfg_entry_point": create_hebi_env(base_env_cfg=base_env, rd_action_class=action_class),
                "rsl_rl_cfg_entry_point": agents.rsl_rl_hebi_agent_cfg.Base_PPORunnerCfg,
                "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
                "d3rlpy_cfg_entry_point": f"{agents.__name__}:d3rlpy_cfg.yaml",
            },
            disable_env_checker=True,
        )


for base_env in decimation_experiments_envs:
    for action_class in action_classes:
        action_class_id = action_class.__name__.replace("RobotActionsCfg_Hebi", "")
        base_env_id = base_env.__name__.replace("_Env", "")
        _id = f"{action_class_id}_{base_env_id}".replace("_", "-")
        gym.register(
            id=_id,
            entry_point="octilab.envs.octi_manager_based_rl:OctiManagerBasedRLEnv",
            kwargs={
                "env_cfg_entry_point": create_hebi_env(base_env_cfg=base_env, rd_action_class=action_class),
                "rsl_rl_cfg_entry_point": agents.rsl_rl_hebi_agent_cfg.DecimationPPORunnerCfg,
                "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
                "d3rlpy_cfg_entry_point": f"{agents.__name__}:d3rlpy_cfg.yaml",
            },
            disable_env_checker=True,
        )


for base_env in strategy3_experiment_envs:
    for action_class in action_classes:
        action_class_id = action_class.__name__.replace("RobotActionsCfg_Hebi", "")
        base_env_id = base_env.__name__.replace("_Env", "")
        _id = f"{action_class_id}_{base_env_id}".replace("_", "-")
        gym.register(
            id=_id,
            entry_point="octilab.envs.octi_manager_based_rl:OctiManagerBasedRLEnv",
            kwargs={
                "env_cfg_entry_point": create_hebi_env(base_env_cfg=base_env, rd_action_class=action_class),
                "rsl_rl_cfg_entry_point": agents.rsl_rl_hebi_agent_cfg.Strategy3ScalePPORunnerCfg,
                "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
                "d3rlpy_cfg_entry_point": f"{agents.__name__}:d3rlpy_cfg.yaml",
            },
            disable_env_checker=True,
        )
        
for base_env in agent_update_rate_experiments_envs:
    for action_class in action_classes:
        action_class_id = action_class.__name__.replace("RobotActionsCfg_Hebi", "")
        base_env_id = base_env.__name__.replace("_Env", "")
        _id = f"{action_class_id}_{base_env_id}".replace("_", "-")
        gym.register(
            id=_id,
            entry_point="octilab.envs.octi_manager_based_rl:OctiManagerBasedRLEnv",
            kwargs={
                "env_cfg_entry_point": create_hebi_env(base_env_cfg=base_env, rd_action_class=action_class),
                "rsl_rl_cfg_entry_point": agents.rsl_rl_hebi_agent_cfg.AgentUpdateRatePPORunnerCfg,
                "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
                "d3rlpy_cfg_entry_point": f"{agents.__name__}:d3rlpy_cfg.yaml",
            },
            disable_env_checker=True,
        )