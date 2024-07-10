# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from d3rlpy."""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse
import numpy as np

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from Stable-Baselines3.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import os
import torch
import traceback

import carb

import octi.lab_tasks.cfgs
import omni.isaac.contrib_tasks  # noqa: F401
import omni.isaac.lab_tasks  # noqa: F401
import math
import d3rlpy
from omni.isaac.lab_tasks.utils import load_cfg_from_registry, parse_env_cfg, get_checkpoint_path
from workflows.d3rlpy.d3rlpy_wrapper import D3rlpyWrapper


class D3Agent:
    def __init__(self, policy, device):
        self.policy = policy
        self.device = device

    def load(self, model_folder, device):
        # load is handled at init
        pass

    # For 1-batch query only!
    def predict(self, sample):

        with torch.no_grad():
            input = torch.from_numpy(sample[0]).float().unsqueeze(0).to("cuda:0")
            at = self.policy(input)[0].to("cpu").detach().numpy()
        return at


def main():
    """Play with stable-baselines agent."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs)
    agent_cfg = load_cfg_from_registry(args_cli.task, "sb3_cfg_entry_point")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    # wrap around environment for stable baselines
    env = D3rlpyWrapper(env, math.inf, np.array([0.02, 0.02, 0.02, 0.1, 0.1, 0.1, 0.1], dtype=np.float32))
    # directory for logging into
    log_root_path = os.path.join("logs", "d3rlpy", args_cli.task)
    log_root_path = os.path.abspath(log_root_path)

    env.seed(seed=agent_cfg["seed"])

    # check checkpoint is valid
    if args_cli.checkpoint is None:
        if args_cli.use_last_checkpoint:
            checkpoint = "model_.*.zip"
        else:
            checkpoint = "model.zip"
        checkpoint_path = get_checkpoint_path(log_root_path, ".*", checkpoint)
    else:
        checkpoint_path = args_cli.checkpoint
    # create agent from stable baselines

    if checkpoint_path is not None:
        policy = torch.jit.load(checkpoint_path)
        policy.to(env.device)
        agent = D3Agent(policy, env.device)
    else:
        raise RuntimeError("No Policy is loaded, please check if correct model is supplied")
    # reset environment
    obs = env.reset()
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = agent.predict([obs])
            # env stepping
            obs, _, _, _ = env.step(actions)

    # close the simulator
    env.close()


if __name__ == "__main__":
    try:
        # run the main execution
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim app
        simulation_app.close()
