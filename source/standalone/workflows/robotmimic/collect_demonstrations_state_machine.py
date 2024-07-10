# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to collect demonstrations with Orbit environments."""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse
from datetime import datetime
from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Collect demonstrations for Orbit environments.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--device", type=str, default="keyboard", help="Device for interacting with environment")
parser.add_argument("--num_demos", type=int, default=1, help="Number of episodes to store in the dataset.")
parser.add_argument("--filename", type=str, default="hdf_dataset", help="Basename of output file.")
parser.add_argument("--convert_delta", type=bool, default=False, help="Convert output of state machine from Absolute to Delta")
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
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml
import omni.isaac.contrib_tasks  # noqa: F401
import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.manipulation.lift import mdp
from omni.isaac.lab_tasks.utils.data_collector import RobomimicDataCollector
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg, load_cfg_from_registry
import tasks
from omni.isaac.lab.utils.math import compute_pose_error, axis_angle_from_quat
from octi.lab_tasks.tasks.craneberryLavaChocoCake.mdp.events import record_state_configuration
def main():
    """Collect demonstrations from the environment using teleop interfaces."""
    # assert (
    #     args_cli.task == "Isaac-Lift-Cube-Franka-IK-Rel-v0"
    # ), "Only 'Isaac-Lift-Cube-Franka-IK-Rel-v0' is supported currently."
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs)

    env_cfg.episode_length_s = 12
    # set the resampling time range to large number to avoid resampling
    # env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
    # we want to have the terms in the observations returned as a dictionary
    # rather than a concatenated tensor
    env_cfg.observations.policy.concatenate_terms = False

    # add termination condition for reaching the goal otherwise the environment won't reset
    # env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    # specify directory for logging experiments
    log_dir = os.path.join("./logs/robomimic", args_cli.task)
    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_dir, date)
    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)

    # create data-collector
    collector_interface = RobomimicDataCollector(
        env_name=args_cli.task,
        directory_path=log_dir,
        filename=args_cli.filename,
        num_demos=args_cli.num_demos,
        flush_freq=env.unwrapped.num_envs,
        env_config={"device": args_cli.device},
    )

    # reset environment
    obs_dict, _ = env.reset()
    collector_interface.reset()
    sm = load_cfg_from_registry(args_cli.task, "state_machine_entry_point")
    sm.initialize(env_cfg.sim.dt * env_cfg.decimation, env.unwrapped.num_envs, env.unwrapped.device)
    # simulate environment -- run everything in inference mode
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while not collector_interface.is_stopped():
            actions = sm.compute(env)
            if args_cli.convert_delta:
                '''Start: Code that translates absolute command to delta command'''
                ee_frame = env.unwrapped.scene.sensors.get("ee_frame")
                ee_pose_b = torch.cat((ee_frame.data.target_pos_w[:, 0, :] - env.unwrapped.scene._default_env_origins, 
                                    ee_frame.data.target_quat_w[:, 0, :]), dim=1).reshape(-1, 7)
                if ee_pose_b[:, 3:].norm() != 0:
                    position_error, quat_angle_error = compute_pose_error(
                            ee_pose_b[:, :3], ee_pose_b[:, 3:], actions[:, :3], actions[:, 3:7], rot_error_type="quat"
                        )
                    pose_error = torch.cat((position_error, quat_angle_error), dim=1)
                else:
                    pose_error = torch.tensor([0, 0, 0, 1.0, 0.0, 0.0, 0.0], device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1)
                
                if env.unwrapped.action_manager.action_term_dim[0] == 6:
                    axis_angles = axis_angle_from_quat(pose_error[:, 3:])
                    pose_error = pose_error[:, :6]
                    pose_error[:, 3:6] = axis_angles[:, :3]

                actions = torch.cat((pose_error, actions[:, -1:]), dim=1)
                '''End: Code that translates absolute command to delta command'''

            # -- obs
            for key, value in obs_dict["policy"].items():
                if key in ['wrist_picture', 'base_picture']:
                    collector_interface.add(f"obs/{key}", value.view(-1, 120, 160, 3))
                else:
                    collector_interface.add(f"obs/{key}", value)
            # -- actions
            collector_interface.add("actions", actions)
            # perform action on environment
            # end_effector_speed = env.unwrapped.data_manager.get_active_term("data", "end_effector_speed")
            # noise = torch.randn_like(actions) * 0.075 * end_effector_speed  # Define noise_stddev based on your action scale
            # noise_action = actions.clone()
            # noise_action[:, :3] += noise[:, :3]

            obs_dict, rewards, terminated, truncated, info = env.step(actions)
            dones = terminated | truncated

            # check that simulation is stopped or not
            if env.unwrapped.sim.is_stopped():
                break
            # robomimic only cares about policy observations
            # store signals from the environment
            # -- next_obs
            for key, value in obs_dict["policy"].items():
                if key in ['wrist_picture', 'base_picture']:
                    collector_interface.add(f"next_obs/{key}", value.view(-1, 120, 160, 3))
                else:
                    collector_interface.add(f"next_obs/{key}", value)
            # -- rewards
            collector_interface.add("rewards", rewards)
            # -- dones
            collector_interface.add("dones", dones)

            # -- is success label
            collector_interface.add("success", env.termination_manager.get_term("success_state"))

            state_record = record_state_configuration(env)
            for asset in state_record.keys():
                for asset_property in state_record[asset]:
                    collector_interface.add(f"record/{asset}-{asset_property}", state_record[asset][asset_property])

            # flush data from collector for successful environments
            reset_env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            collector_interface.flush(reset_env_ids)
            if(dones.any()):
                sm.reset_idx(dones.nonzero(as_tuple=False).squeeze(-1))

    # close the simulator
    collector_interface.close()
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
