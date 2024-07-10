
"""Script to train RL agent with RSL-RL."""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Pick and lift state machine for cabinet environments.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest everything else."""

import gymnasium as gym
import torch

import traceback
from omni.isaac.lab_tasks.utils import parse_env_cfg

import carb

import omni.isaac.lab_tasks  # noqa: F401
import tasks  # noqa: F401
from omni.isaac.lab_tasks.manager_based.manipulation.cabinet.cabinet_env_cfg import CabinetEnvCfg
from omni.isaac.lab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg
from omni.isaac.lab.utils.math import axis_angle_from_quat, compute_pose_error, euler_xyz_from_quat
from octi.lab_tasks.tasks.craneberryLavaChocoCake.config.hebi.state_machines.cranberry_on_cake import CranberryDecoratorSm


def main():
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        use_gpu=not args_cli.cpu,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.episode_length_s = 13
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    sm = load_cfg_from_registry(args_cli.task, "state_machine_entry_point")
    sm.initialize(env_cfg.sim.dt * env_cfg.decimation, env.unwrapped.num_envs, env.unwrapped.device)
    # reset environment at start
    env.reset()
    # create state machine
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            actions = sm.compute(env)
            # end_effector_speed = env.unwrapped.data_manager.get_active_term("data", "end_effector_speed")
            # noise = torch.randn_like(actions) * 0.075 * end_effector_speed  # Define noise_stddev based on your action scale
            # actions[:, :3] += noise[:, :3]

            # ee_frame = env.scene.sensors.get("ee_frame")
            # ee_pose_b = torch.cat((ee_frame.data.target_pos_w[:, 0, :] - env.unwrapped.scene._default_env_origins, 
            #                        ee_frame.data.target_quat_w[:, 0, :]), dim=1).reshape(-1, 7)
            # if ee_pose_b[:, 3:].norm() != 0:
            #     position_error, quat_angle_error = compute_pose_error(
            #             ee_pose_b[:, :3], ee_pose_b[:, 3:], actions[:, :3], actions[:, 3:7], rot_error_type="quat"
            #         )
            #     pose_error = torch.cat((position_error, quat_angle_error), dim=1)
            # else:
            #     pose_error = torch.tensor([0, 0, 0, 1.0, 0.0, 0.0, 0.0], device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1)
            
            # if env.action_manager.action_term_dim[0] == 6:
            #     axis_angles = axis_angle_from_quat(pose_error[:, 3:])
            #     pose_error = pose_error[:, :6]
            #     pose_error[:, 3:6] = axis_angles[:, :3]

            # actions = torch.cat((pose_error, actions[:, -1:]), dim=1)
            # step environment
            obs, _, reset_terminated, reset_time_outs, _ = env.step(actions)
            done = reset_terminated | reset_time_outs
            # reset state machine
            if done.any():
                sm.reset_idx(done.nonzero(as_tuple=False).squeeze(-1))

    # close the environment
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
