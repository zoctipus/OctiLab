# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch
from collections.abc import Sequence
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab.envs import ManagerBasedRLEnv
from .deformable_base_env import DeformableBaseEnv
from omni.isaac.lab.envs.manager_based_env import VecEnvObs
from .octi_manager_based_rl_cfg import OctiManagerBasedRLEnvCfg
from ..managers.data_manager import DataManager
VecEnvStepReturn = tuple[VecEnvObs, torch.Tensor, torch.Tensor, torch.Tensor, dict]


class OctiManagerBasedRLEnv(ManagerBasedRLEnv):
    """The superclass for reinforcement learning-based environments.

    This class inherits from :class:`ManagerBasedEnvCfg` and implements the core functionality for
    reinforcement learning-based environments. It is designed to be used with any RL
    library. The class is designed to be used with vectorized environments, i.e., the
    environment is expected to be run in parallel with multiple sub-environments. The
    number of sub-environments is specified using the ``num_envs``.

    Each observation from the environment is a batch of observations for each sub-
    environments. The method :meth:`step` is also expected to receive a batch of actions
    for each sub-environment.

    While the environment itself is implemented as a vectorized environment, we do not
    inherit from :class:`gym.vector.VectorEnv`. This is mainly because the class adds
    various methods (for wait and asynchronous updates) which are not required.
    Additionally, each RL library typically has its own definition for a vectorized
    environment. Thus, to reduce complexity, we directly use the :class:`gym.Env` over
    here and leave it up to library-defined wrappers to take care of wrapping this
    environment for their agents.

    Note:
        For vectorized environments, it is recommended to **only** call the :meth:`reset`
        method once before the first call to :meth:`step`, i.e. after the environment is created.
        After that, the :meth:`step` function handles the reset of terminated sub-environments.
        This is because the simulator does not support resetting individual sub-environments
        in a vectorized environment.

    """
    cfg: OctiManagerBasedRLEnvCfg
    """Configuration for the environment."""

    def __init__(self, cfg: OctiManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        ManagerBasedRLEnv.__bases__ = (DeformableBaseEnv, gym.Env)
        super().__init__(cfg, render_mode)
        # RLTaskEnv.__bases__ = (BaseEnv, gym.Env)
        frame_marker_cfg = FRAME_MARKER_CFG.copy()
        frame_marker_cfg.markers["frame"].scale = (0.02, 0.02, 0.02)
        self.goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))
        self.extensions = {}
    """
    Operations - MDP
    """

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics and reset terminated environments.

        Unlike the :class:`ManagerBasedEnvCfg.step` class, the function performs the following operations:

        1. Process the actions.
        2. Perform physics stepping.
        3. Perform rendering if gui is enabled.
        4. Update the environment counters and compute the rewards and terminations.
        5. Reset the environments that terminated.
        6. Compute the observations.
        7. Return the observations, rewards, resets and extras.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        # process actions
        self.action_manager.process_action(action)
        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            render = self._sim_step_counter % self.cfg.sim.render_interval == 0 and (
                self.sim.has_gui() or self.sim.has_rtx_sensors()
            )
            # simulate
            self.sim.step(render=render)
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # des_pos = self.action_manager._terms.get('index_finger')._ik_controller
        # self.goal_marker.visualize(
        #     des_pos.ee_pos_des[0] + self.scene._default_env_origins, des_pos.ee_quat_des[0, :, 0:4])
        if getattr(self.cfg, "data_manager", None) is not None:
            self.data_manager.compute()
        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute()
        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def load_managers(self):
        if getattr(self.cfg, "datas", None) is not None:
            self.data_manager: DataManager = DataManager(self.cfg.datas, self)
            print("[INFO] Data Manager: ", self.data_manager)
        super().load_managers()

    def _reset_idx(self, env_ids: Sequence[int]):
        if getattr(self, "data_manager", None) is not None:
            self.data_manager.reset(env_ids)
        return super()._reset_idx(env_ids)

    def close(self):
        for key, val in self.extensions.items():
            del val
        super().close()
