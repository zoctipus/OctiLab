# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper to configure an :class:`ManagerBasedRLEnv` instance to Diversity Skill vectorized environment.

"""

from __future__ import annotations

import gym.spaces
import gymnasium
import gym
import torch
import math
import numpy as np
from omni.isaac.lab.envs import ManagerBasedRLEnv


class DiversitySkillWrapper:
    """Wraps around Orbit environment for D3rlpy library

    It is Crucial for d3rlpy to have only 1 environment in training. This wrapper assume rl-Env returned
    action and observation are in shape (1, {}_space), it will transform it to ({}_space)

    """

    def __init__(self, env: ManagerBasedRLEnv, actions_bounds):
        """Initializes the wrapper.

        Note:
            The wrapper calls :meth:`reset` at the start since the Df runner does not call reset.

        Args:
            env: The environment to wrap around.

        Raises:
            ValueError: When the environment is not an instance of :class:`ManagerBasedRLEnv`.
        """
        # check that input is valid
        if not isinstance(env.unwrapped, ManagerBasedRLEnv):
            raise ValueError(f"The environment must be inherited from ManagerBasedRLEnv. Environment type: {type(env)}")
        # initialize the wrapper
        self.env = env
        # store information required by wrapper
        self.action_bounds = actions_bounds
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length
        self.num_actions = self.unwrapped.action_manager.total_action_dim
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
        policy_obs_space = self.unwrapped.observation_space["policy"]
        return policy_obs_space

    @property
    def action_space(self) -> gym.Space:
        # # """Returns the :attr:`Env` :attr:`action_space`."""
        # return self.env.action_space

        """Returns the :attr:`Env` :attr:`action_space`."""
        # note: rl-games only wants single action space
        action_space = self.unwrapped.action_space
        return gym.spaces.Box(self.action_bounds[0], self.action_bounds[1], action_space.shape)

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
        return obs

    def step(self, actions:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # record step information
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        dones = (terminated | truncated).to(dtype=torch.long)
        # move extra observations to the extras dict
        obs = obs_dict["policy"]
        extras["observations"] = obs_dict
        # move time out information to the extras dict
        # this is only needed for infinite horizon tasks
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        # return the step information
        return obs, rew, dones, extras

    def close(self):  # noqa: D102
        return self.env.close()
