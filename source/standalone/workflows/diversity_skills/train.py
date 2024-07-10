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

import octi.lab_tasks.cfgs
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


def train(env, agent: DSACAgent, args_cli, agent_cfg, logger, p_z, device):
    num_envs = env.num_envs
    n_steps_per_env = agent_cfg["n_steps_per_env"]
    n_updates_per_iteration = 1
    if args_cli.checkpoint is not None:
        episode, total_sample_step, last_logq_zs, rng_states = logger.load_weights(
            args_cli.checkpoint, load_replay_buffer=True
        )
        agent.hard_update_all_target_networks()
        min_episode = episode
        env.seed(agent_cfg["seed"])
        print("Keep training from previous run.")

    else:
        min_episode = 0
        total_sample_step = 0
        last_logq_zs = 0
        np.random.seed(agent_cfg["seed"])
        env.seed(agent_cfg["seed"])
        env.observation_space.seed(agent_cfg["seed"])
        env.action_space.seed(agent_cfg["seed"])
        print("Training from scratch.")

    """
    Random Exploration
    """
    z = np.random.choice(agent_cfg["n_skills"], p=p_z)
    state = env.reset()
    state = concat_state_latent_batch(state, z, agent_cfg["n_skills"])
    step = 0
    episode_reward = 0
    while total_sample_step < agent_cfg["random_explore_steps"]:
        action = env.action_space.sample()
        z = np.random.choice(agent_cfg["n_skills"], p=p_z)
        next_state, reward, done, _ = env.step(action)
        next_state = concat_state_latent_batch(next_state, z, agent_cfg["n_skills"])
        agent.vec_store(state, z, done, action, next_state, reward)
        state = next_state
        step += 1
        episode_reward += reward
        if done or step > n_steps_per_env:
            total_sample_step += step
            z = np.random.choice(agent_cfg["n_skills"], p=p_z)
            state = env.reset()
            state = concat_state_latent_batch(state, z, agent_cfg["n_skills"])
            step = 0
            print("Collected a trajectory using randomly sampled action, rew =", episode_reward)
            episode_reward = 0

    """
    Sampling from Env and Training
    """
    rewbuffer = deque(maxlen=100)
    lenbuffer = deque(maxlen=100)
    cur_reward_sum = torch.zeros(num_envs, dtype=torch.float, device=device)
    cur_episode_length = torch.zeros(num_envs, dtype=torch.float, device=device)
    frames_per_iteration = n_steps_per_env * num_envs
    episode_completed = 0
    ep_infos = []

    state = env.reset()
    state = concat_state_latent_batch(state, z, agent_cfg["n_skills"])
    for it in tqdm(range(1, agent_cfg["max_n_iterations"] + 1)):
        z = np.random.choice(agent_cfg["n_skills"], p=p_z)
        start = time.time()
        n_sampling_steps = (
            agent_cfg["batch_size"] if (len(agent.vec_memory) < agent_cfg["batch_size"]) else n_steps_per_env
        )
        for i in range(n_sampling_steps):
            action = agent.vec_choose_action(state)
            next_state, reward, done, infos = env.step(action)
            next_state = concat_state_latent_batch(next_state, z, agent_cfg["n_skills"])

            state_t = state.to("cpu")
            next_state_t = next_state.to("cpu")
            action_t = action.to("cpu")
            reward_t = reward.to("cpu")
            done_t = done.to("cpu")
            z_t = torch.zeros((state_t.shape[0]), device="cpu")

            agent.vec_store(state_t, z_t, done_t, action_t, next_state_t, reward_t)
            state = next_state
            cur_reward_sum += reward
            cur_episode_length += 1
            new_ids = (done > 0).nonzero(as_tuple=False)
            rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
            lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
            episode_completed += len(new_ids)
            cur_reward_sum[new_ids] = 0
            cur_episode_length[new_ids] = 0

            if "log" in infos:
                ep_infos.append(infos["log"])
        stop = time.time()
        collection_time = stop - start
        total_sample_step += frames_per_iteration
        losses_list = []

        # OSINAYN does not add diversity reward UNTIL after the policy is close to the optimal
        if len(rewbuffer) > 0:
            mean_episode_reward = statistics.mean(rewbuffer)
            train_with_diversity = mean_episode_reward > agent_cfg["reward_epsilon"]
        else:
            train_with_diversity = False
        learn_time_start = time.time()
        n_updates_per_iteration = max(int(it / agent_cfg["max_n_iterations"] * n_steps_per_env), 5)

        train_sampling_time, train_training_time = 0, 0
        critic_sampling_time, critic_training_time = 0, 0
        for _ in range(n_steps_per_env):
            losses = agent.train(diversity_reward=train_with_diversity)
            train_sampling_time += losses["train_sampling_time"]
            train_training_time += losses["train_training_time"]
            losses_list.append(losses)

            for _ in range(agent_cfg["utd"] - 1):
                critic_train_info = agent.train_critic(diversity_reward=train_with_diversity)
                critic_sampling_time += critic_train_info["critic_sampling_time"]
                critic_training_time += critic_train_info["critic_training_time"]

        learn_time_stop = time.time()
        learn_time = learn_time_stop - learn_time_start

        if len(losses_list):
            loss_dict = {k: np.average([dic[k] for dic in losses_list]) for k in losses_list[0].keys()}
            bufferinfo = agent.vec_memory
            logger.log(
                it,
                frames_per_iteration,
                episode_completed,
                rewbuffer,
                lenbuffer,
                collection_time,
                learn_time,
                train_sampling_time,
                train_training_time,
                critic_sampling_time,
                critic_training_time,
                bufferinfo.time_spent_permutation,
                bufferinfo.time_spent_flattening,
                bufferinfo.time_spent_moving,
                len(agent.vec_memory),
                agent.vec_memory.usage_percentage(),
                ep_infos,
                z,
                loss_dict,
                total_sample_step,
                np.random.get_state(),
            )
            logger.log_train(loss_dict, total_sample_step)
        ep_infos.clear()

    logger._save_weights(episode_completed, total_sample_step, *agent.get_rng_states())


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
    action_bounds = [env.action_space.low[0], env.action_space.high[0]]

    agent_cfg.update({"n_envs": n_envs, "n_states": n_states, "n_actions": n_actions, "action_bounds": action_bounds})
    print("agent_cfg:", agent_cfg)

    p_z = np.full(agent_cfg["n_skills"], 1 / agent_cfg["n_skills"])
    agent = DSACAgent(
        p_z, agent_cfg, replay_buffer_device="cpu", replay_buffer_loading_path=args_cli.buffer_loading_path
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

    agent.logger = logger

    if agent_cfg["do_train"]:
        train(env, agent, args_cli, agent_cfg, logger, p_z, env.device)


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
