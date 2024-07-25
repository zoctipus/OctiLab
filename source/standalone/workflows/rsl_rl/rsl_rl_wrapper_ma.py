# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper to configure an :class:`ManagerBasedRLEnv` instance to RSL-RL vectorized environment.

The following example shows how to wrap an environment for RSL-RL:

.. code-block:: python

    from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper

    env = RslRlVecEnvWrapper(env)

"""


import gymnasium as gym
import torch
from multiprocessing import Barrier, Array, Lock
from rsl_rl.env import VecEnv

from omni.isaac.lab.envs import DirectRLEnv, ManagerBasedRLEnv


class WorkerEnv(VecEnv):
    def __init__(self, master_env, process_index):
        super(WorkerEnv, self).__init__()
        self.master_env = master_env
        self.process_index = process_index
        self.env_idx = self.master_env.env_partitions[process_index]

    @property
    def process_idx(self) -> str | None:
        """Returns the :attr:`process_index`."""
        return self.process_index

    @property
    def render_mode(self) -> str | None:
        """Returns the :attr:`Env` :attr:`render_mode`."""
        return self.master_env.render_mode

    @property
    def observation_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`observation_space`."""
        return self.master_env.observation_space

    @property
    def action_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`action_space`."""
        return self.master_env.env_terrains

    @property
    def num_envs(self):
        return len(self.env_idx)

    @property
    def num_actions(self):
        return self.master_env.num_actions

    @property
    def episode_length_buf(self):
        return self.master_env.episode_length_buf[self.env_idx]

    def get_observations(self) -> tuple[list[torch.Tensor], dict]:
        """Returns the current observations of the environment."""
        return self.master_env.get_observations(self.process_index)

    def reset(self):
        return self.master_env.reset()

    def step(self, action):
        # Send the action and process index to the shared environment
        self.master_env.step((action, self.process_index))

        # Retrieve the result for this process
        return self.master_env.get_results(self.process_index)


class RslRlVecEnvWrapperMa(VecEnv):
    """Wraps around Isaac Lab environment for RSL-RL library

    To use asymmetric actor-critic, the environment instance must have the attributes :attr:`num_privileged_obs` (int).
    This is used by the learning agent to allocate buffers in the trajectory memory. Additionally, the returned
    observations should have the key "critic" which corresponds to the privileged observations. Since this is
    optional for some environments, the wrapper checks if these attributes exist. If they don't then the wrapper
    defaults to zero as number of privileged observations.

    .. caution::

        This class must be the last wrapper in the wrapper chain. This is because the wrapper does not follow
        the :class:`gym.Wrapper` interface. Any subsequent wrappers will need to be modified to work with this
        wrapper.

    Reference:
        https://github.com/leggedrobotics/rsl_rl/blob/master/rsl_rl/env/vec_env.py
    """

    def __init__(self, env: ManagerBasedRLEnv):
        """Initializes the wrapper.

        Note:
            The wrapper calls :meth:`reset` at the start since the RSL-RL runner does not call reset.

        Args:
            env: The environment to wrap around.

        Raises:
        """
        # check that input is valid
        if not isinstance(env.unwrapped, ManagerBasedRLEnv) and not isinstance(env.unwrapped, DirectRLEnv):
            raise ValueError(
                "The environment must be inherited from ManagerBasedRLEnv or DirectRLEnv. Environment type:"
                f" {type(env)}"
            )
        # initialize the wrapper
        self.env = env
        # store information required by wrapper
        self.num_envs = [len(partition) for partition in self.env_partitions]
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length
        if hasattr(self.unwrapped, "action_manager"):
            self.num_actions = self.unwrapped.action_manager.total_action_dim
        else:
            self.num_actions = self.unwrapped.num_actions
        if hasattr(self.unwrapped, "observation_manager"):
            self.num_obs = self.unwrapped.observation_manager.group_obs_dim["policy"][0]
        else:
            self.num_obs = self.unwrapped.num_observations
        # -- privileged observations
        if (
            hasattr(self.unwrapped, "observation_manager")
            and "critic" in self.unwrapped.observation_manager.group_obs_dim
        ):
            self.num_privileged_obs = self.unwrapped.observation_manager.group_obs_dim["critic"][0]
        elif hasattr(self.unwrapped, "num_states"):
            self.num_privileged_obs = self.unwrapped.num_states
        else:
            self.num_privileged_obs = 0
        # reset at the start since the RSL-RL runner does not call reset
        _, reset_extra = self.env.reset()
        self.observations = [None for _ in range(len(self.env_partitions))]
        self.actions = [None for _ in range(len(self.env_partitions))]
        self.rewards = [None for _ in range(len(self.env_partitions))]
        self.terminated = [None for _ in range(len(self.env_partitions))]
        self.truncated = [None for _ in range(len(self.env_partitions))]
        self.dones = [None for _ in range(len(self.env_partitions))]
        self.extras = [{} for _ in range(len(self.env_partitions))]
        for extra in self.extras:
            extra.update(reset_extra)

    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    """
    Properties -- Gym.Wrapper
    """

    @property
    def cfg(self) -> object:
        """Returns the configuration class instance of the environment."""
        return self.unwrapped.cfg

    @property
    def render_mode(self) -> str | None:
        """Returns the :attr:`Env` :attr:`render_mode`."""
        return self.env.render_mode

    @property
    def observation_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`observation_space`."""
        return self.env.observation_space

    @property
    def action_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`action_space`."""
        return self.env.action_space

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__

    @property
    def unwrapped(self) -> ManagerBasedRLEnv:
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped

    @property
    def env_partitions(self) -> list[torch.Tensor]:
        return self.env.unwrapped.scene.env_partitions
    """
    Properties
    """

    def get_observations(self, index=None) -> tuple[list[torch.Tensor], dict]:
        """Returns the current observations of the environment."""
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()
        if index is None:
            obs = [obs_dict["policy"][i] for i in self.env_partitions]
        else:
            obs_dict["policy"] = obs_dict["policy"][self.env_partitions[index]]
            obs = obs_dict["policy"]
        return obs, {"observations": obs_dict}

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """The episode length buffer."""
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        """Set the episode length buffer.

        Note:
            This is needed to perform random initialization of episode lengths in RSL-RL.
        """
        self.unwrapped.episode_length_buf = value

    """
    Operations - MDP
    """

    def seed(self, seed: int = -1) -> int:  # noqa: D102
        return self.unwrapped.seed(seed)

    def reset(self) -> tuple[torch.Tensor, dict]:  # noqa: D102
        # reset the environment
        obs_dict, _ = self.env.reset()
        # return observations
        return obs_dict["policy"], {"observations": obs_dict}

    def step(self, actions: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], dict]:
        actions = torch.cat(actions, dim=0)
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        # compute dones for compatibility with RSL-RL
        # dones = (terminated | truncated).to(dtype=torch.long)
        # move extra observations to the extras dict
        partitions = self.env_partitions
        for idx, partition in enumerate(partitions):
            self.observations[idx] = obs_dict["policy"][partition]
            self.rewards[idx] = rew[partition]
            self.terminated[idx] = terminated[partition]
            self.truncated[idx] = truncated[partition]
            self.dones[idx] = (self.terminated[idx] | self.truncated[idx]).to(dtype=torch.long)
            self.extras[idx]["observations"] = {"policy": self.observations[idx]}
            if not self.unwrapped.cfg.is_finite_horizon:
                self.extras[idx]["time_outs"] = self.truncated[idx]

        # extras["observations"] = obs_dict
        # move time out information to the extras dict
        # this is only needed for infinite horizon tasks
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        # return the step information
        return self.observations, self.rewards, self.dones, self.extras

    def close(self):  # noqa: D102
        return self.env.close()

    def get_results(self, process_index):
        self.barrier.wait()  # Wait for all processes to finish stepping
        obs = self.observations[process_index]
        rewards = self.rewards[process_index]
        dones = self.dones[process_index]
        extras = self.extras[process_index]
        self.barrier.wait()  # Synchronize after reading
        return obs, rewards, dones, extras
