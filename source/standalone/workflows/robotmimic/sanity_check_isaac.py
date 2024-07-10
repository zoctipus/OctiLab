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
parser.add_argument("--checkpoint", type=str, default=None, help="Pytorch model checkpoint to load.")
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
from omni.isaac.lab.envs.mdp.events import reset_root_state_uniform
import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import parse_env_cfg
from omni.isaac.lab.managers import SceneEntityCfg

def set_init_state_brute_force(env, hacked_obs_init):
    
    
    envids = torch.arange(env.num_envs, device = env.device)
    cake_pos_b_init = hacked_obs_init['cake_pos_b'][0]
    canberry_pos_b_init = hacked_obs_init['canberry_pos_b'][0]
    canberry_default_pos = env.scene['canberry'].data.default_root_state[envids][0]
    cake_default_pos = env.scene['cake'].data.default_root_state[envids][0]
    cake_offset = 5 * [0.0002, -0.0014, 0.0215]
    cake_x = cake_pos_b_init[0]-cake_default_pos[0] - cake_offset[0]
    cake_y = cake_pos_b_init[1]-cake_default_pos[1] - cake_offset[1]
    cake_z = cake_pos_b_init[2]-cake_default_pos[2] - cake_offset[2]
    reset_root_state_uniform(env, envids,{
                                "x":(cake_x, cake_x), 
                                "y":(cake_y, cake_y), 
                                "z":(cake_z, cake_z)
                            }, {}, SceneEntityCfg("cake", body_names="cake"))
    reset_root_state_uniform(env, envids,{
                                "x":(canberry_pos_b_init[0]-canberry_default_pos[0], canberry_pos_b_init[0]-canberry_default_pos[0]), 
                                "y":(canberry_pos_b_init[1]-canberry_default_pos[1], canberry_pos_b_init[1]-canberry_default_pos[1]), 
                                "z":(canberry_pos_b_init[2]-canberry_default_pos[2], canberry_pos_b_init[2]-canberry_default_pos[2])
                            }, {}, SceneEntityCfg("canberry", body_names="canberry"))
    env.sim.step(render=False)
    # update buffers at sim dt
    env.scene.update(dt=env.physics_dt)
    env.sim.render()

def main():
    """Run a trained policy from robomimic with Orbit environment."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=1, use_fabric=not args_cli.disable_fabric)
    # we want to have the terms in the observations returned as a dictionary
    # rather than a concatenated tensor
    env_cfg.observations.policy.concatenate_terms = False
    # env_cfg.episode_length_s = 19
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # acquire device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    # restore policy
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=args_cli.checkpoint, device=device, verbose=True)

    # reset environment
    obs_dict, _ = env.reset()
    # robomimic only cares about policy observations
    obs = obs_dict["policy"]
    # simulate environment
    count = 0
    dataset_observation = {}
    true_action = []
    while simulation_app.is_running():
        with h5py.File("logs/robomimic/IkDeltaDls-ImplicitMotorHebi-JointPos-CraneberryLavaChocoCake-v0/2024-04-15_16-02-41/hdf_dataset.hdf5", "r") as f:
            for demo_key in f[f"mask/successful_valid"]:
                demo_key_string = demo_key.decode()
                # demo_key_string = 'demo_107'
                observation_data = f[f"data/{demo_key_string}/obs"]
                for key in observation_data:
                    dataset_observation[key] = torch.from_numpy(f[f"data/{demo_key_string}/obs/{key}"][:]).to(env.device)
                true_action = torch.from_numpy(f[f"data/{demo_key_string}/actions"][:]).to(env.device)
                error_sum = torch.zeros(true_action.shape[1], device = env.device)
                error_percentage_sum = torch.zeros(true_action.shape[1], device = env.device)
                error_percentage_average = torch.zeros(true_action.shape[1], device = env.device)
                hacked_single_obs = {"robot_actions":[], "robot_joint_pos":[], "robot_joint_vel":[], "robot_eepose":[], "canberry_pos_b":[], "cake_pos_b":[]}
                for key, value in dataset_observation.items():
                    hacked_single_obs[key] = value[2].view(1, -1)
                obs_diff = {"robot_actions":0, "robot_joint_pos":0, "robot_joint_vel":0, "robot_eepose":0, "canberry_pos_b":0, "cake_pos_b":0}
                obs = env.reset()[0]["policy"]
                set_init_state_brute_force(env, hacked_single_obs)
                for i in range(len(true_action) - 1):
                # run everything in inference mode
                    with torch.inference_mode():
                        # hacked_obs = {}
                        for key, value in dataset_observation.items():
                            hacked_single_obs[key] = value[i].view(1, -1)
                        # compute actions
                        for key, value in dataset_observation.items():
                            if key in ["canberry_pos_b", "cake_pos_b", "robot_eepose"]:
                                diff = torch.norm(obs[key] - hacked_single_obs[key], dim=1)
                                obs_diff[key] += diff
                        predicted_action = policy(obs)

                        # predicted_action = torch.from_numpy(predicted_action).to(device=device).view(1, env.action_space.shape[1])
                        predicted_action = torch.from_numpy(predicted_action).to(device=device)
                        # action_quat = torch.tensor([0.0, 0.0, 1.0, 0.0], device = env.device)
                        # action_full = torch.cat((predicted_action[:3], action_quat, predicted_action[-1].view(1)), dim=0).view(1, -1)
                        error = torch.abs(predicted_action - true_action[i].view(1, -1))
                        error_percentage = error / true_action[i].view(1, -1)
                        error_sum += error[0]
                        error_percentage_sum += error_percentage[0].clamp(max=1)
                        error_percentage_average = (error_percentage_sum/count).clamp(max=1)
                        # apply predicted_action
                        obs_dict = env.step(predicted_action.view(1, -1))[0]
                        # obs_dict = env.step(true_action[i].view(1, -1))[0]
                        # robomimic only cares about policy observations
                        obs = obs_dict["policy"]
                        count += 1
                        # print(error_percentage_average)
                        print(error[:, :3])
                

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
