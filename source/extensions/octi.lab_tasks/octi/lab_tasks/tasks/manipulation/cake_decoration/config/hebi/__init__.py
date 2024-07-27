# from .hebi_cfg import HEBI_IMPLICITY_ACTUATOR_CFG, HEBI_IDEAL_PD_CFG, HEBI_PWM_MOTOR_CFG
# from .env_cfg import *
# from .rope_env_cfg import *
# from .rope_env_cfg_controller import *
# from .Hebi_JointPos_CraneberryLavaChocoCake_Env import *
from .tycho_joint_pos import *
from .state_machines.cranberry_on_cake import CranberryDecoratorSm
from octi.lab.envs.create_env import create_hebi_env
import gymnasium as gym
import os
from . import agents, tycho_joint_pos


gym.register(
    id="Octi-Cake-Decoration-Tycho-IkDel-v0",
    entry_point="octi.lab.envs.octi_manager_based_rl:OctiManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": tycho_joint_pos.CakeDecorationTychoIkdelta,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Base_PPORunnerCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
        "d3rlpy_cfg_entry_point": f"{agents.__name__}:d3rlpy_cfg.yaml",
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc.json"),  # type: ignore
        "robomimic_bcq_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bcq.json"),  # type: ignore
        "state_machine_entry_point": CranberryDecoratorSm,
        "diversity_skill_entry_point": f"{agents.__name__}:diversity_skill_cfg.yaml",
    },
    disable_env_checker=True,
)


gym.register(
    id="Octi-Cake-Decoration-Tycho-IkAbs-v0",
    entry_point="octi.lab.envs.octi_manager_based_rl:OctiManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": tycho_joint_pos.CakeDecorationTychoIkabsolute,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Base_PPORunnerCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
        "d3rlpy_cfg_entry_point": f"{agents.__name__}:d3rlpy_cfg.yaml",
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc.json"),  # type: ignore
        "robomimic_bcq_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bcq.json"),  # type: ignore
        "state_machine_entry_point": CranberryDecoratorSm,
        "diversity_skill_entry_point": f"{agents.__name__}:diversity_skill_cfg.yaml",
    },
    disable_env_checker=True,
)


gym.register(
    id="Octi-Cake-Decoration-Tycho-JointPos-v0",
    entry_point="octi.lab.envs.octi_manager_based_rl:OctiManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": tycho_joint_pos.CakeDecorationTychoJointPosition,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Base_PPORunnerCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
        "d3rlpy_cfg_entry_point": f"{agents.__name__}:d3rlpy_cfg.yaml",
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc.json"),  # type: ignore
        "robomimic_bcq_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bcq.json"),  # type: ignore
        "state_machine_entry_point": CranberryDecoratorSm,
        "diversity_skill_entry_point": f"{agents.__name__}:diversity_skill_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Octi-Cake-Decoration-Tycho-JointEff-v0",
    entry_point="octi.lab.envs.octi_manager_based_rl:OctiManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": tycho_joint_pos.CakeDecorationTychoJointEffort,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Base_PPORunnerCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
        "d3rlpy_cfg_entry_point": f"{agents.__name__}:d3rlpy_cfg.yaml",
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc.json"),  # type: ignore
        "robomimic_bcq_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bcq.json"),  # type: ignore
        "state_machine_entry_point": CranberryDecoratorSm,
        "diversity_skill_entry_point": f"{agents.__name__}:diversity_skill_cfg.yaml",
    },
    disable_env_checker=True,
)