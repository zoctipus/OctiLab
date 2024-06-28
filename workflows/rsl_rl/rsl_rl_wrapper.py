from omni.isaac.lab.envs.manager_based_rl_env import ManagerBasedRLEnv
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper
import torch


class RslActionClipWrapper(RslRlVecEnvWrapper):

    def __init__(self, env: ManagerBasedRLEnv):
        super().__init__(env)

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # clip the action between -1.0 to 1.0.
        actions = torch.tanh(actions)
        return super().step(actions)
