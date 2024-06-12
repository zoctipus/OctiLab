# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a trained policy from robomimic."""

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
parser.add_argument("--dataset", type=str, default=None, help="The dataset to replay in simulation")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import h5py
import robomimic  # noqa: F401
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import omni.isaac.contrib_tasks  # noqa: F401
import tasks
import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import parse_env_cfg


def main():
    """Run a trained policy from robomimic with Orbit environment."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=1, use_fabric=not args_cli.disable_fabric)
    # we want to have the terms in the observations returned as a dictionary
    # rather than a concatenated tensor
    env_cfg.observations.policy.concatenate_terms = False

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # reset environment
    obs_dict, _ = env.reset()
    # robomimic only cares about policy observations
    obs = obs_dict["policy"]
    # simulate environment
    count = 0
    observation = {}
    action = []
    while simulation_app.is_running():
        with h5py.File(args_cli.dataset, "r") as f:
            # for demo_key in f["data"].items():
            for demo_key in f["mask/successful"]:
                demo_key_string = demo_key.decode()
                observation_data = f[f"data/{demo_key_string}/obs"]
                # observation_data = f[f"data/{demo_key[0]}/obs"]
                for key in observation_data:
                    observation[key] = torch.from_numpy(f[f"data/{demo_key_string}/obs/{key}"][:]).to(env.device)
                action = torch.from_numpy(f[f"data/{demo_key_string}/actions"][:]).to(env.device)
                
                for i in range(len(action) - 1):
                # run everything in inference mode
                    with torch.inference_mode():
                        hacked_obs = {}
                        for key, value in observation.items():
                            hacked_obs[key] = value[i].view(1, -1)
                        # action_quat = torch.tensor([0.0, 0.0, 1.0, 0.0], device = env.device)
                        # action_full = torch.cat((action[i][:3], action_quat, action[i][-1].view(1)), dim=0).view(1, -1)
                        # apply actions
                        obs_dict = env.step(action[i].view(1, -1))[0]
                        # obs_dict = env.step(action_full)[0]
                        # robomimic only cares about policy observations
                        obs = obs_dict["policy"]
                        count += 1
                env.reset()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
