# from .hebi_cfg import HEBI_IMPLICITY_ACTUATOR_CFG, HEBI_IDEAL_PD_CFG, HEBI_PWM_MOTOR_CFG
# from .env_cfg import *
# from .rope_env_cfg import *
# from .rope_env_cfg_controller import *
# from .Hebi_JointPos_CraneberryLavaChocoCake_Env import *
from .Hebi_JointPos_CraneberryLavaChocoCake_Env import *
from .state_machines.cranberry_on_cake import CranberryDecoratorSm
from octilab.envs.create_env import create_hebi_env
import gymnasium as gym
import os
from . import agents


base_envs = [PwmMotorHebi_JointPos_CraneberryLavaChocoCake_Env, 
             IdealPDHebi_JointPos_CraneberryLavaChocoCake_Env, 
             ImplicitMotorHebi_JointPos_CraneberryLavaChocoCake_Env]

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
                "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc.json"),
                "robomimic_bcq_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bcq.json"),
                "state_machine_entry_point": CranberryDecoratorSm,
                "diversity_skill_entry_point": f"{agents.__name__}:diversity_skill_cfg.yaml",
                "d3rlpy_cfg_entry_point": f"{agents.__name__}:d3rlpy_cfg.yaml"
            },
            disable_env_checker=True,
        )