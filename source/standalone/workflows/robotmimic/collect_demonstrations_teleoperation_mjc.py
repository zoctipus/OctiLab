# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to collect demonstrations with Orbit environments."""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Collect demonstrations for Orbit environments.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--device", type=str, default="keyboard", help="Device for interacting with environment")
parser.add_argument("--num_demos", type=int, default=1, help="Number of episodes to store in the dataset.")
parser.add_argument("--filename", type=str, default="hdf_dataset", help="Basename of output file.")
parser.add_argument("--save", type=bool, default=False, help="save camera render")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import gymnasium as gym
import os
import torch
from omni.isaac.lab.devices import Se3Keyboard, Se3SpaceMouse
from devices.se3_keyboard import Se3KeyboardAbsolute 
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

import omni.isaac.contrib_tasks  # noqa: F401
import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.manipulation.lift import mdp
from omni.isaac.lab_tasks.utils.data_collector import RobomimicDataCollector
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg

import omni.replicator.core as rep
from omni.isaac.lab.sensors.camera import Camera
from omni.isaac.lab.utils import convert_dict_to_backend
from omni.isaac.lab.sim import SimulationCfg, SimulationContext
from tycho_env import TychoEnv,TychoEnv_glasspot, TychoEnv_child_clock, TychoEnv_cake, TychoEnv_cherry, TychoEnv_pick,TychoEnv_reach, TychoEnv_coin, TychoEnv_push, TychoEnv_needle, TychoEnv_box,TychoEnv_clock, TychoEnv_glass

def pre_process_actions(delta_pose: torch.Tensor, gripper_command: bool) -> torch.Tensor:
    """Pre-process actions for the environment."""
    # compute actions based on environment
    if "Reach" in args_cli.task:
        # note: reach is the only one that uses a different action space
        # compute actions
        return delta_pose
    else:
        # resolve gripper command
        gripper_vel = torch.zeros((delta_pose.shape[0], 1), dtype=torch.float, device=delta_pose.device)
        gripper_vel[:] = -1 if gripper_command else 1
        # compute actions
        return torch.concat([delta_pose, gripper_vel], dim=1)


def main():
    """Collect demonstrations from the environment using teleop interfaces."""
    env = TychoEnv_glasspot(
        config = {
                "state_space": "eepose-obj",
            }
        )
    env.curriculum = 1000000000
    env._max_episode_steps = 1000000
    torch_device = "cuda:0" if torch.cuda.is_available() else "cpu"
     # create controller
    if args_cli.device.lower() == "keyboard":
        teleop_interface = Se3KeyboardAbsolute(pos_sensitivity=0.003, rot_sensitivity=0.02)
    elif args_cli.device.lower() == "spacemouse":
        teleop_interface = Se3SpaceMouse(pos_sensitivity=0.05, rot_sensitivity=0.005)
    else:
        raise ValueError(f"Invalid device interface '{args_cli.device}'. Supported: 'keyboard', 'spacemouse'.")
    # add teleoperation key for env reset
    sim_cfg = SimulationCfg(dt=0.01, substeps=1)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # Play the simulator
    sim.reset()
    teleop_interface.add_callback("L", env.reset)
    # print helper
    print(teleop_interface)

    # specify directory for logging experiments
    log_dir = os.path.join("./logs/robomimic", args_cli.task)

    # create data-collector
    collector_interface = RobomimicDataCollector(
        env_name=args_cli.task,
        directory_path=log_dir,
        filename=args_cli.filename,
        num_demos=args_cli.num_demos,
        flush_freq=1,
        env_config={"device": args_cli.device},
    )

    # reset environment
    obs_array = torch.tensor(env.reset(), device=torch_device).unsqueeze(0)
    # reset interfaces
    teleop_interface.reset()
    collector_interface.reset()
    success_counter = 0
    last_action = torch.zeros(8, device=torch_device)
    # simulate environment -- run everything in inference mode
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while not collector_interface.is_stopped():
            # get keyboard command
            delta_pose, gripper_command = teleop_interface.advance()
            # convert to torch
            delta_pose = torch.tensor(delta_pose, dtype=torch.float, device=torch_device).repeat(1, 1)
            # compute actions based on environment
            actions = pre_process_actions(delta_pose, gripper_command)
            save_data = not torch.equal(actions, last_action)
            last_action = actions.clone()
            # perform action on environment
            actions[:, :3] -= obs_array[:, :3]
            next_obs_array, rewards, dones, info = env.step(actions[0].cpu().numpy())
            
            sim.step()
            env.render()
            next_obs_array = torch.tensor(next_obs_array, device=torch_device).unsqueeze(0)
            rewards = torch.tensor(rewards, device=torch_device).unsqueeze(0)
            dones = torch.tensor(dones, dtype=torch.int, device=torch_device).unsqueeze(0)
            successes = torch.tensor(info['success'], dtype=torch.int, device=torch_device).unsqueeze(0)
            # robomimic only cares about policy observations
            # store signals from the environment
            # -- next_obs
            if save_data or successes[0]:
                # -- observation
                collector_interface.add("obs", obs_array)
                # -- action
                collector_interface.add("actions", actions)
                # -- next observation
                collector_interface.add("next_obs", next_obs_array)
                # -- rewards
                collector_interface.add("rewards", rewards)
                # -- dones
                collector_interface.add("dones", dones)
                # -- is success label
                collector_interface.add("success", successes)
                # flush data from collector for successful environments
            
            if successes[0]:
                success_counter +=1
            success_dones = dones | successes
            
            reset_env_ids = success_dones.nonzero(as_tuple=False).squeeze(-1)
            if success_counter > 10:
                collector_interface.flush(reset_env_ids)
            else:
                collector_interface.flush(torch.tensor([], device=torch_device))
            if(dones[0] or success_counter > 10):
                teleop_interface.reset()
                env.reset()
                success_counter = 0
            
            obs_array = next_obs_array.clone()
                
                
    # close the simulator
    collector_interface.close()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
