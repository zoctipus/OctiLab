# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a keyboard teleoperation with Orbit manipulation environments."""

from __future__ import annotations
import argparse
from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation for Orbit environments.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--device", type=str, default="keyboard", help="Device for interacting with environment")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")

"""Launch Isaac Sim Simulator first."""
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import carb
from omni.isaac.lab.devices import Se3Gamepad, Se3SpaceMouse
from octilab.devices import Se3KeyboardAbsolute, RokokoGlove, RokokoGloveKeyboard
import omni.isaac.lab_tasks  # noqa: F401
import ext.envs.envs.tasks  # noqa: F401
import ext.envs.envs.tasks.manipulations  # noqa: F401
from omni.isaac.lab_tasks.utils import parse_env_cfg
from omni.isaac.lab.utils.math import quat_mul, quat_from_angle_axis
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
# from omni.isaac.lab.markers import VisualizationMarkers


def pre_process_actions(delta_pose: torch.Tensor, gripper_command: bool) -> torch.Tensor:
    """Pre-process actions for the environment."""
    # compute actions based on environment
    d_pose = delta_pose.clone()
    if "Reach" in args_cli.task:
        # note: reach is the only one that uses a different action space
        # compute actions
        return d_pose
    else:
        # resolve gripper command
        gripper_vel = torch.zeros(d_pose.shape[0], 1, device=d_pose.device)
        gripper_vel[:] = -1.0 if gripper_command else 1.0
        # compute actions
        return torch.concat([d_pose, gripper_vel], dim=1)


def pre_process_glove_actions(absolute_pose: torch.Tensor, placeholder_command: bool) -> torch.Tensor:
    device = absolute_pose.device
    absolute_pose[:, 3:] = absolute_pose[:, [6, 3, 4, 5]]
    absolute_pose[:, :3] = absolute_pose[:, [0, 2, 1]]
    absolute_pose[:, 2] += 0.5
    rot_actions = torch.tensor([[0.0, 0.0, 1.57]], device=device)
    angle: torch.Tensor = torch.linalg.vector_norm(rot_actions, dim=1)
    axis = rot_actions / angle.unsqueeze(-1)
    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    rot_delta_quat = torch.where(
        angle.unsqueeze(-1).repeat(1, 4) > 1.0e-6, quat_from_angle_axis(angle, axis), identity_quat
    ).repeat(len(absolute_pose), 1)
    absolute_pose[:, 3:] = quat_mul(absolute_pose[:, 3:], rot_delta_quat)
    return absolute_pose[:, :3].reshape(1, -1)


def main():
    """Running keyboard teleoperation with Orbit manipulation environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    # modify configuration
    env_cfg.terminations.time_out = None  # type: ignore
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    # check environment name (for reach , we don't allow the gripper)
    if "Reach" in args_cli.task:
        carb.log_warn(
            f"The environment '{args_cli.task}' does not support gripper control. The device command will be ignored."
        )

    # create controller
    if args_cli.device.lower() == "keyboard":
        teleop_interface = Se3KeyboardAbsolute(
            pos_sensitivity=0.01 * args_cli.sensitivity, rot_sensitivity=0.01 * args_cli.sensitivity
        )
        def preprocess_func(x): return pre_process_actions(*x)  # noqa: E704
    elif args_cli.device.lower() == "spacemouse":
        teleop_interface = Se3SpaceMouse(
            pos_sensitivity=0.05 * args_cli.sensitivity, rot_sensitivity=0.01 * args_cli.sensitivity
        )
        def preprocess_func(x): return pre_process_actions(*x)  # noqa: E704
    elif args_cli.device.lower() == "gamepad":
        teleop_interface = Se3Gamepad(
            pos_sensitivity=0.1 * args_cli.sensitivity, rot_sensitivity=0.1 * args_cli.sensitivity
        )
        def preprocess_func(x): return pre_process_actions(*x)  # noqa: E704
    elif args_cli.device.lower() == "rokoko_smartglove":
        teleop_interface = RokokoGlove(
            UDP_IP="0.0.0.0", UDP_PORT=14043, scale=1.65,
            right_hand_track=["rightHand", "rightIndexMedial", "rightMiddleMedial", "rightRingMedial",
                              "rightThumbTip", "rightIndexTip", "rightMiddleTip", "rightRingTip"]
        )
        def preprocess_func(x): return pre_process_glove_actions(*x)  # noqa: E704, E306
    elif args_cli.device.lower() == "rokoko_smartglove_keyboard":
        teleop_interface = RokokoGloveKeyboard(
            pos_sensitivity=0.1 * args_cli.sensitivity, rot_sensitivity=0.1 * args_cli.sensitivity,
            UDP_IP="0.0.0.0", UDP_PORT=14043, scale=1.65,
            right_hand_track=["rightHand", "rightIndexMedial", "rightMiddleMedial", "rightRingMedial",
                              "rightThumbTip", "rightIndexTip", "rightMiddleTip", "rightRingTip"]
        )
        def preprocess_func(x): return pre_process_glove_actions(*x)  # noqa: E704, E306
    else:
        raise ValueError(f"Invalid device interface '{args_cli.device}'.\
            Supported: 'keyboard', 'spacemouse', 'rokoko_smartgloves'.")
    # add teleoperation key for env reset
    teleop_interface.add_callback("L", env.reset)
    # print helper for keyboard
    print(teleop_interface)

    frame_marker_cfg = FRAME_MARKER_CFG.copy()  # type: ignore
    frame_marker_cfg.markers["frame"].scale = (0.02, 0.02, 0.02)
    # goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))
    # reset environment
    env.reset()
    teleop_interface.reset()
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # get keyboard command
            teleop_output = teleop_interface.advance()
            # test = teleop_interface.debug_advance_all_joint_data()[0]
            # test[:, 3:] = test[:, [6, 3, 4, 5]]
            # test[:, :3] = test[:, [0, 2, 1]]
            # test[:, 2] += 0.5
            # rot_actions = torch.tensor([[0.0, 0.0, 1.57]], device=env.unwrapped.device)
            # angle: torch.Tensor = torch.linalg.vector_norm(rot_actions, dim=1)
            # axis = rot_actions / angle.unsqueeze(-1)
            # identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.unwrapped.device)
            # rot_delta_quat = torch.where(
            #     angle.unsqueeze(-1).repeat(1, 4) > 1.0e-6, quat_from_angle_axis(angle, axis), identity_quat
            # ).repeat(len(test), 1)
            # test[:, 3:] = quat_mul(test[:, 3:], rot_delta_quat)
            # goal_marker.visualize(
            #     test[:, :3] + env.unwrapped.scene._default_env_origins, test[:, 3:])
            actions_clone = preprocess_func(teleop_output)
            # print(actions_clone)
            env.step(actions_clone)
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
