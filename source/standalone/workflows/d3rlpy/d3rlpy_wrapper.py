# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper to configure an :class:`ManagerBasedRLEnv` instance to RSL-RL vectorized environment.

The following example shows how to wrap an environment for RSL-RL:

.. code-block:: python

    from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper

    env = RslRlVecEnvWrapper(env)

"""

from __future__ import annotations

import gymnasium
import gym
import torch
import math
import numpy as np
from omni.isaac.lab.envs import ManagerBasedEnv


class D3rlpyWrapper:
    """Wraps around Orbit environment for D3rlpy library

    It is Crucial for d3rlpy to have only 1 environment in training. This wrapper assume rl-Env returned
    action and observation are in shape (1, {}_space), it will transform it to ({}_space)

    """

    def __init__(self, env: ManagerBasedEnv, clip_obs: float, clip_actions: float):
        """Initializes the wrapper.

        Note:
            The wrapper calls :meth:`reset` at the start since the Df runner does not call reset.

        Args:
            env: The environment to wrap around.

        Raises:
            ValueError: When the environment is not an instance of :class:`ManagerBasedRLEnv`.
        """
        # check that input is valid
        if not isinstance(env.unwrapped, ManagerBasedEnv):
            raise ValueError(f"The environment must be inherited from ManagerBasedRLEnv. Environment type: {type(env)}")
        # initialize the wrapper
        self.env = env
        self.clip_actions = clip_actions
        # store information required by wrapper
        self.num_envs = self.unwrapped.num_envs
        if (self.num_envs != 1) :
            raise ValueError("To use d3rlpy, num_env must equal 1, your num_env is {self.num_envs}")
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length
        self.num_actions = self.unwrapped.action_manager.total_action_dim
        self.num_obs = self.unwrapped.observation_manager.group_obs_dim["policy"][0]
        # -- privileged observations
        if "critic" in self.unwrapped.observation_manager.group_obs_dim:
            self.num_privileged_obs = self.unwrapped.observation_manager.group_obs_dim["critic"][0]
        else:
            self.num_privileged_obs = 0
        # reset at the start since the RSL-RL runner does not call reset
        self.env.reset()

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
        # note: rl-games only wants single observation space
        policy_obs_space = self.unwrapped.single_observation_space["policy"]
        if not isinstance(policy_obs_space, gymnasium.spaces.Box):
            raise NotImplementedError(
                f"The RL-Games wrapper does not currently support observation space: '{type(policy_obs_space)}'."
                f" If you need to support this, please modify the wrapper: {self.__class__.__name__},"
                " and if you are nice, please send a merge-request."
            )
        # note: maybe should check if we are a sub-set of the actual space. don't do it right now since
        #   in ManagerBasedRLEnv we are setting action space as (-inf, inf).
        return gym.spaces.Box(-math.inf, math.inf, policy_obs_space.shape)

    @property
    def action_space(self) -> gym.Space:
        # # """Returns the :attr:`Env` :attr:`action_space`."""
        # return self.env.action_space

        """Returns the :attr:`Env` :attr:`action_space`."""
        # note: rl-games only wants single action space
        action_space = self.unwrapped.single_action_space
        if not isinstance(action_space, gymnasium.spaces.Box):
            raise NotImplementedError(
                f"The RL-Games wrapper does not currently support action space: '{type(action_space)}'."
                f" If you need to support this, please modify the wrapper: {self.__class__.__name__},"
                " and if you are nice, please send a merge-request."
            )
        # return casted space in gym.spaces.Box (OpenAI Gym)
        return gym.spaces.Box(-self.clip_actions, self.clip_actions, action_space.shape)

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__

    @property
    def unwrapped(self) -> ManagerBasedEnv:
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped

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
        obs = obs_dict["policy"]
        if len(obs) != 1 or len(obs.shape) != 2:
            raise ValueError(f"to use d3rlpy you can only train 1 env at a time, you observation needs to\
                             have (1, n), but your shape is {obs.shape} ")
        return obs[0].cpu().numpy()

    def step(self, actions: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # record step information
        actions_torch = torch.from_numpy(actions).unsqueeze(0).to(self.env.device)
        actions_scaled = torch.tanh(actions_torch) * 0.03
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions_scaled)
        # compute dones for compatibility with RSL-RL
        dones = (terminated | truncated).to(dtype=torch.long)
        # move extra observations to the extras dict
        obs = obs_dict["policy"]
        if len(obs) != 1 or len(obs.shape) != 2:
            raise ValueError(f"to use d3rlpy you can only train 1 env at a time, you observation needs to\
                             have (1, n), but your shape is {obs.shape} ")
        obs = obs[0].cpu().numpy()
        rew = rew[0].cpu().numpy()
        extras["observations"] = obs_dict
        # move time out information to the extras dict
        # this is only needed for infinite horizon tasks
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        # return the step information
        return obs, rew, dones, extras

    def close(self):  # noqa: D102
        return self.env.close()
