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
    # assert (
    #     args_cli.task == "Isaac-Lift-Cube-Franka-IK-Rel-v0"
    # ), "Only 'Isaac-Lift-Cube-Franka-IK-Rel-v0' is supported currently."
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs)

    # modify configuration such that the environment runs indefinitely
    # until goal is reached
    env_cfg.terminations.time_out = None
    # set the resampling time range to large number to avoid resampling
    # env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
    # we want to have the terms in the observations returned as a dictionary
    # rather than a concatenated tensor
    env_cfg.observations.policy.concatenate_terms = False

    # add termination condition for reaching the goal otherwise the environment won't reset
    # env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # create controller
    if args_cli.device.lower() == "keyboard":
        teleop_interface = Se3KeyboardAbsolute(pos_sensitivity=0.003, rot_sensitivity=0.02)
    elif args_cli.device.lower() == "spacemouse":
        teleop_interface = Se3SpaceMouse(pos_sensitivity=0.05, rot_sensitivity=0.005)
    else:
        raise ValueError(f"Invalid device interface '{args_cli.device}'. Supported: 'keyboard', 'spacemouse'.")
    # add teleoperation key for env reset
    teleop_interface.add_callback("L", env.reset)
    # print helper
    print(teleop_interface)

    # specify directory for logging experiments
    log_dir = os.path.join("./logs/robomimic", args_cli.task)
    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)

    # create data-collector
    collector_interface = RobomimicDataCollector(
        env_name=args_cli.task,
        directory_path=log_dir,
        filename=args_cli.filename,
        num_demos=args_cli.num_demos,
        flush_freq=env.num_envs,
        env_config={"device": args_cli.device},
    )

    # reset environment
    obs_dict, _ = env.reset()

    # reset interfaces
    teleop_interface.reset()
    collector_interface.reset()

    # extract entities for simplified notation
    camera_wrist: Camera = env.scene["camera_wrist"]
    camera_base: Camera = env.scene["camera_base"]
    # Create replicator writer
    
    rootpath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
    wrist_output_dir = os.path.join(rootpath, log_dir[2:], "camera_wrist")
    base_output_dir = os.path.join(rootpath, log_dir[2:], "camera_base")
    rep_writer_wrist = rep.BasicWriter(
        output_dir=wrist_output_dir,
        frame_padding=0,
        colorize_instance_id_segmentation=camera_wrist.cfg.colorize_instance_id_segmentation,
        colorize_instance_segmentation=camera_wrist.cfg.colorize_instance_segmentation,
        colorize_semantic_segmentation=camera_wrist.cfg.colorize_semantic_segmentation,
    )

    rep_writer_base = rep.BasicWriter(
        output_dir=base_output_dir,
        frame_padding=0,
        colorize_instance_id_segmentation=camera_wrist.cfg.colorize_instance_id_segmentation,
        colorize_instance_segmentation=camera_wrist.cfg.colorize_instance_segmentation,
        colorize_semantic_segmentation=camera_wrist.cfg.colorize_semantic_segmentation,
    )
    camera_index = 0

    # simulate environment -- run everything in inference mode
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while not collector_interface.is_stopped():
            # get keyboard command
            delta_pose, gripper_command = teleop_interface.advance()
            # convert to torch
            delta_pose = torch.tensor(delta_pose, dtype=torch.float, device=env.device).repeat(env.num_envs, 1)
            # compute actions based on environment
            actions = pre_process_actions(delta_pose, gripper_command)

            # TODO: Deal with the case when reset is triggered by teleoperation device.
            #   The observations need to be recollected.
            # store signals before stepping
            # -- obs
            for key, value in obs_dict["policy"].items():
                collector_interface.add(f"obs/{key}", value)
            # -- actions
            collector_interface.add("actions", actions)
            # perform action on environment
            obs_dict, rewards, terminated, truncated, info = env.step(actions)
            dones = terminated | truncated

            camera_wrist.update(dt=env.unwrapped.sim.get_physics_dt())
            camera_base.update(dt=env.unwrapped.sim.get_physics_dt())
            if args_cli.save:
                # Save images from camera at camera_index
                # note: BasicWriter only supports saving data in numpy format, so we need to convert the data to numpy.
                # tensordict allows easy indexing of tensors in the dictionary
                wrist_cam_data = convert_dict_to_backend(camera_wrist.data.output[camera_index], backend="numpy")

                # Extract the other information
                wrist_cam_info = camera_wrist.data.info[camera_index]

                # Pack data back into replicator format to save them using its writer
                rep_output_wrist = dict()
                for key, data, info in zip(wrist_cam_data.keys(), wrist_cam_data.values(), wrist_cam_info.values()):
                    if info is not None:
                        rep_output_wrist[key] = {"data": data, "info": info}
                    else:
                        rep_output_wrist[key] = data
                # Save images
                # Note: We need to provide On-time data for Replicator to save the images.
                rep_output_wrist["trigger_outputs"] = {"on_time": camera_base.frame[camera_index]}
                rep_writer_wrist.write(rep_output_wrist)

                # Save images from camera at camera_index
                # note: BasicWriter only supports saving data in numpy format, so we need to convert the data to numpy.
                # tensordict allows easy indexing of tensors in the dictionary
                base_cam_data = convert_dict_to_backend(camera_base.data.output[camera_index], backend="numpy")

                # Extract the other information
                base_cam_info = camera_wrist.data.info[camera_index]

                # Pack data back into replicator format to save them using its writer
                rep_output_base = dict()
                for key, data, info in zip(base_cam_data.keys(), base_cam_data.values(), base_cam_info.values()):
                    if info is not None:
                        rep_output_base[key] = {"data": data, "info": info}
                    else:
                        rep_output_base[key] = data
                # Save images
                # Note: We need to provide On-time data for Replicator to save the images.
                rep_output_base["trigger_outputs"] = {"on_time": camera_base.frame[camera_index]}
                rep_writer_base.write(rep_output_base)

            # check that simulation is stopped or not
            if env.unwrapped.sim.is_stopped():
                break
            # robomimic only cares about policy observations
            # store signals from the environment
            # -- next_obs
            for key, value in obs_dict["policy"].items():
                collector_interface.add(f"next_obs/{key}", value)
            # -- rewards
            collector_interface.add("rewards", rewards)
            # -- dones
            collector_interface.add("dones", dones)

            # -- is success label
            collector_interface.add("success", env.termination_manager.get_term("success_state"))

            # flush data from collector for successful environments
            reset_env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            collector_interface.flush(reset_env_ids)
            if(dones[0]):
                teleop_interface.reset()

    # close the simulator
    collector_interface.close()
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
