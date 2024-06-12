# Octi Added
"""Script to train RL agent with D3rlpy.
"""
"""Launch Isaac Sim Simulator first."""
import argparse
from omni.isaac.lab.app import AppLauncher
# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with D3rlpy.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--load_demo", type=str, default=None, help="Path to demo.")
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
from datetime import datetime
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from workflows.d3rlpy.d3rlpy_wrapper import D3rlpyWrapper
from workflows.d3rlpy.hdf5ToNpy import hdf5_to_npy
import d3rlpy
import math
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    """Train with D3rlpy agent."""
    # parse configuration
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs)
    agent_cfg = load_cfg_from_registry(args_cli.task, "d3rlpy_cfg_entry_point")

    # override configuration with command line arguments
    if args_cli.seed is not None:
        agent_cfg["seed"] = args_cli.seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "d3rlpy", args_cli.task)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_root_path, log_name)
    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for stable baselines
    # if env.action_space.shape
    env = D3rlpyWrapper(env, math.inf, 1.0)
    env.seed(seed=agent_cfg["seed"])
    d3rlpy.seed(agent_cfg["seed"])
 
    sac = d3rlpy.algos.SAC(
        use_gpu=True,
        actor_learning_rate=agent_cfg["actor_learning_rate"],
        critic_learning_rate=agent_cfg["critic_learning_rate"],
        temp_learning_rate=agent_cfg["temp_learning_rate"],
        dropout=agent_cfg["dropout"],
        layernorm=agent_cfg["layernorm"]
    )

    if args_cli.checkpoint is not None:
        print("loading  ", args_cli.checkpoint)
        sac.build_with_env(env)
        sac.load_model(args_cli.checkpoint)
    buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=agent_cfg["replay_buffer_max_length"], env=env)

    # load demo path
    demo_loading_path = None
    if args_cli.load_demo is not None:
        demo_loading_path = os.path.join(log_dir, "demo.npy")
        if args_cli.load_demo.endswith("hdf5"):
            hdf5_to_npy(args_cli.load_demo, demo_loading_path, env.unwrapped.observation_manager.active_terms['policy'])
        elif args_cli.load_demo.endswith("npy"):
            demo_loading_path = args_cli.load_demo
        else:
            raise ValueError("load_demo_path of none hdf5 or npy file is not supported")
        print("")
    # start training
    sac.fit_online(
        env, buffer, n_steps=agent_cfg["n_steps"], n_steps_per_epoch=agent_cfg["n_steps_per_epoch"], \
        random_steps=agent_cfg["random_steps"], logdir=log_dir, eval_env=env, tensorboard_dir=log_dir+'/tensorboard', \
        save_interval=agent_cfg["save_interval"], utd=agent_cfg["utd"], load_demo=demo_loading_path)

    dataset = buffer.to_mdp_dataset()
    dataset.dump(os.path.join(log_dir, "dataset.h5"))
    sac.save_policy(os.path.join(log_dir, 'policy.pt'))
    sac.save_model(os.path.join(log_dir, 'model.pt'))
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
