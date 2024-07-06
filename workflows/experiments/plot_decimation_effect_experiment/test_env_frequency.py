# simulate.py
import argparse
import gymnasium as gym
import torch
import numpy as np
from omni.isaac.lab.app import AppLauncher


def eval_env(simulation_app, args_cli, env_cfg):
    """Running keyboard teleoperation with Orbit manipulation environment."""
    # modify configuration
    env_cfg.terminations.time_out = None  # type: ignore
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    # check environment name (for reach , we don't allow the gripper)
    env.reset()
    count = 0
    positions = []
    applied_torque = []
    joint_pos_target = []
    current_pos_target = []
    count = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # get keyboard command
            actions_clone = torch.tensor(
                [[-0.5000, -0.3000, 0.0500, -1.5497e-05, 2.0993e-05, 7.1197e-01, -7.0221e-01, 1]],
                device=env.unwrapped.device,
            )
            # print(actions_clone)
            env.step(actions_clone)
            joint_pos_target.append(env.unwrapped.scene["robot"].data.joint_pos_target.cpu().numpy())
            current_pos_target.append(env.unwrapped.scene["robot"].data.joint_pos.cpu().numpy())
            applied_torque.append(env.unwrapped.scene["robot"].data.applied_torque.clone())
            positions.append(env.unwrapped.scene["robot"].data.body_pos_w[0, -1, :3].cpu().numpy())
            count += 1
            if count >= 1000:  # Change this to your desired stopping condition
                break
    # close the simulator
    env.close()
    return positions, joint_pos_target, current_pos_target


def main():
    parser = argparse.ArgumentParser(description="Keyboard teleoperation for Orbit environments.")
    parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
    parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument("--decimation", type=int, required=True, help="Simulation decimation value.")

    # launch omniverse app
    AppLauncher.add_app_launcher_args(parser)
    # parse the arguments
    args_cli = parser.parse_args()
    app_launcher = AppLauncher(headless=args_cli.headless)
    simulation_app = app_launcher.app

    import omni.isaac.lab_tasks  # noqa: F401
    import ext.envs.envs.tasks  # noqa: F401
    import ext.envs.envs.tasks.manipulations  # noqa: F401
    from omni.isaac.lab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(
        args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_update_dt = env_cfg.sim.dt * env_cfg.decimation
    env_cfg.decimation = args_cli.decimation
    env_cfg.sim.dt = agent_update_dt / env_cfg.decimation

    positions, joint_pos_target, current_pos_target = eval_env(simulation_app, args_cli, env_cfg)
    positions = np.array(positions)
    joint_pos_target = np.array(joint_pos_target)
    np.save(f'logs/test/positions_decimation_{args_cli.decimation}.npy', positions)
    np.save(f'logs/test/joint_pos_target_{args_cli.decimation}.npy', joint_pos_target)
    np.save(f'logs/test/current_pos_target_{args_cli.decimation}.npy', current_pos_target)


if __name__ == "__main__":
    main()
