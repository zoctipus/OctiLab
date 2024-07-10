# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play policy trained using robomimic for Orbit environments.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="number of environments")
parser.add_argument("--checkpoint", type=str, default=None, help="Pytorch model checkpoint to load.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()


"""Rest everything follows."""

import gymnasium as gym
import torch

import robomimic  # noqa: F401
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
from tycho_env import TychoEnv, TychoEnv_cherry, TychoEnv_pick,TychoEnv_reach, TychoEnv_coin, TychoEnv_push, TychoEnv_needle, TychoEnv_box,TychoEnv_clock, TychoEnv_glass



def main():
    """Run a trained policy from robomimic with Orbit environment."""
    env_mjc = TychoEnv_clock(config = {
                "action_space":"eepose-delta",
                "ee_workspace_low": [-0.5,-0.5,0.02,0,-1,0,0,-0.6],               # lock the orientation
                "ee_workspace_high":[-0.2,-0.1,0.08,0,-1,0,0,-0.18],             # lock the orientation
                "action_low": [-0.2, -0.2, -0.2, 0,0,0,0, -0.5],                 # lock the orientation
                "action_high": [0.2, 0.2, 0.2, 1e-10,1e-10,1e-10,1e-10, 0.5],        # lock the orientation
                "state_space": "eepose-obj-clock",
                "dr":False,
                "reset_eepose": [-0.30, -0.30, 0.05,0,-1,0,0, -0.36],
                "normalized_action": False,
                'static_ball':True,
                'sample_points':True,
                "out_dir":"logs/mujoco"
            })
    # acquire device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    # restore policy
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=args_cli.checkpoint, device=device, verbose=True)

    # reset environment
    obs = env_mjc.reset()
    # simulate environment
    while True:
        # run everything in inference mode
        with torch.inference_mode():
            # compute actions
            actions = policy({"state": torch.tensor(obs, device=device).unsqueeze(0)})
            obs, reward, done, info = env_mjc.step(actions)
            env_mjc.render()
            if (done):
                obs = env_mjc.reset()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
