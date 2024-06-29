# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse
from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
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
parser.add_argument(
    "--buffer_loading_path", type=str, default=None, help="adding demostration experience to boost learning"
)
parser.add_argument("--checkpoint", type=str, default=None, help="loading the checkpoint")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import os
import traceback
from datetime import datetime

import carb
from collections import deque
from stable_baselines3.common.callbacks import CheckpointCallback

from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

import ext.envs.cfgs
import omni.isaac.contrib_tasks  # noqa: F401
import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import load_cfg_from_registry, parse_env_cfg

import math
import numpy as np
import torch
import time
from tqdm import tqdm
import statistics
from workflows.diversity_skills.diversity_skill_wrapper import DiversitySkillWrapper
from workflows.diversity_skills.Brain.vec_agent_os import DSACAgent
from workflows.diversity_skills.Common import Logger


def set_seed(seed):
    seed = int(seed)
    import random

    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)


def concat_state_latent(s, z_, n):
    z_one_hot = np.zeros(n)
    z_one_hot[z_] = 1
    return np.concatenate([s, z_one_hot])


def concat_state_latent_batch(s, z_, n):
    z_one_hot = torch.zeros((s.shape[0], n), device=s.device)
    z_one_hot[:, z_] = 1
    return torch.cat([s, z_one_hot], axis=1)


def main():
    """Train with Diversity Skill agent."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs)
    agent_cfg = load_cfg_from_registry(args_cli.task, "diversity_skill_entry_point")

    # # override configuration with command line arguments
    if args_cli.seed is not None:
        agent_cfg["seed"] = args_cli.seed
    log_dir = os.path.join("logs", "diversity_skills", args_cli.task, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
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
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = DiversitySkillWrapper(env, [-1, 1])

    set_seed(agent_cfg["seed"])
    env.seed(agent_cfg["seed"])

    n_envs = env.num_envs
    n_states = env.observation_space.shape[1:]
    n_actions = env.action_space.shape[1:]
    action_bounds = [env.action_space.low[1:], env.action_space.high[1:]]

    agent_cfg.update({"n_envs": n_envs, "n_states": n_states, "n_actions": n_actions, "action_bounds": action_bounds})
    print("agent_cfg:", agent_cfg)

    p_z = np.full(agent_cfg["n_skills"], 1 / agent_cfg["n_skills"])
    agent = DSACAgent(
        p_z, agent_cfg, replay_buffer_device="cuda:0", replay_buffer_loading_path=args_cli.buffer_loading_path
    )
    logger = Logger(agent, agent_cfg, False, device="cuda")

    if agent_cfg["datafile"]:
        with open(agent_cfg["datafile"], "rb") as f:
            episodes = np.load(f, allow_pickle=True)
            for episode in episodes:
                z = np.random.choice(agent_cfg["n_skills"], p=p_z)
                obs, act, rew, done = episode["obs"], episode["act"], episode["rew"], episode["done"]
                obs_z = concat_state_latent_batch(obs, z, agent_cfg["n_skills"])
                for t in range(len(obs) - 1):
                    agent.vec_store(obs_z[t], z, done[t], act[t], obs_z[t + 1], rew[t])
    logger.load_weights(model_path=args_cli.checkpoint, load_replay_buffer=False)
    agent.logger = logger

    # reset environment
    obs = env.reset()
    z = torch.tensor(0, device=env.device)
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            obs = concat_state_latent_batch(obs, z, agent_cfg["n_skills"])
            # agent stepping
            actions = agent.choose_deterministic_action(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)

    # close the simulator
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
