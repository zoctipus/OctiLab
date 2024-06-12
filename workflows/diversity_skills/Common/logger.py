import time
import psutil
from torch.utils.tensorboard import SummaryWriter
import torch
import os
import numpy as np
import datetime
import glob
import json
import statistics

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class Logger:
    def __init__(self, agent, config, verbose, device = "cpu"):
        self.config = config
        self.device = device
        self.verbose = verbose
        self.agent = agent
        self.log_dir = "logs/diversity_skills/"+ self.config["env_name"][:-3] + "/" + (self.config["agent_name"] + "/" if self.config["agent_name"] != "" else "") + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.checkpoint_dir = os.path.join(self.log_dir, "checkpoints")
        self.start_time = 0
        self.duration = 0
        self.running_logq_zs = 0
        self.max_episode_reward = -torch.inf
        self._turn_on = False
        self.to_gb = lambda in_bytes: in_bytes / 1024 / 1024 / 1024
        self.skill_episode_counter = {}
        self.total_time = 0

        if self.config["do_train"]:
            self.writer = SummaryWriter(self.log_dir)
            self.write_params()
            self._log_params()
            if not os.path.exists(self.checkpoint_dir):
                os.mkdir(self.checkpoint_dir)

    def write_params(self):
        with open(os.path.join(self.log_dir, "params.txt"), 'w') as param_fn:
            print("Writing params")
            print(json.dumps(self.config, cls=NumpyEncoder))
            print(json.dumps(self.config, cls=NumpyEncoder), file=param_fn)

    def _log_params(self):
        for k, v in self.config.items():
            self.writer.add_text(k, str(v))

    def log(self, it, frames_per_iteration, episode_completed, rewbuffer, lenbuffer, 
            collection_time, learn_time, train_sampling_time, train_training_time, critic_sampling_time, critic_training_time,
            time_spent_permutation, time_spent_flattening, time_spent_moving, buffer_transitions, usage_percentage,
            ep_infos, skill, loss_dict, total_sample_step, *rng_states, width = 80, pad = 35):

        logq_zs = -loss_dict['Loss/discriminator_loss'] # log q(z|s)
        single_iteration_time = collection_time + learn_time
        self.total_time += single_iteration_time
        self.running_logq_zs = 0.99 * self.running_logq_zs + 0.01 * logq_zs if self.running_logq_zs != 0 else logq_zs
        
        if len(rewbuffer) > 0:
            self.max_episode_reward = max(self.max_episode_reward, max(rewbuffer))
            mean_episode_reward = statistics.mean(rewbuffer)
        else:
            mean_episode_reward = 0

        if it % self.config["interval"] == 0:
            print("saving models")
            idx = str(int(it / self.config["interval"]))
            self._save_weights(episode_completed, total_sample_step, idx, *rng_states)

        ep_string = ""
        if ep_infos:
            for key in ep_infos[0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in ep_infos:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, it)
                    if self.verbose:
                        ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, it)
                    if self.verbose:
                        ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        self.writer.add_scalar("Rewards/MaxEpisodeReward", self.max_episode_reward, episode_completed)
        self.skill_episode_counter[skill] = 0 if skill not in self.skill_episode_counter else self.skill_episode_counter[skill] + 1
        self.writer.add_scalar(f"Rewards/Skill{str(skill)}Reward", mean_episode_reward, self.skill_episode_counter[skill])

        self.writer.add_scalar("Loss/Running logq(z|s)", self.running_logq_zs, total_sample_step)
        self.writer.add_scalar("Eval/Total Sample Traj", episode_completed, total_sample_step)

        self.writer.add_histogram(f"Histogram/Skill{str(skill)}Reward", mean_episode_reward)
        self.writer.add_histogram("Histogram/AllRewards", mean_episode_reward)

        fps = int(frames_per_iteration / single_iteration_time)
        string = f" \033[1m Learning iteration {it}/{self.config['max_n_iterations']} \033[0m "
        usage_percentage *= 100
        extra_string = ""
        log_string = (
                f"""{'#' * width}\n"""
                f"""{string.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {collection_time:.3f}s, learning {learn_time:.3f}s)\n"""
                f"""{'':>{pad}} {"":>{14}}(replay sample :{(train_sampling_time + critic_sampling_time):.3f}s, train {(train_training_time):.3f}s, critic {(critic_training_time):.3f}s)\n"""
                f"""{'Replay Buffer:':>{pad}} {usage_percentage:.3f}% usage,  {buffer_transitions:.0f} transitions, {time_spent_permutation:.3f}s permute, {time_spent_flattening:.3f}s flat, {time_spent_moving:.3f}s move\n"""
                f"""{'Policy loss:':>{pad}} {loss_dict["Loss/policy_loss"]:.4f}\n"""
                f"""{'Q loss:':>{pad}} {loss_dict["Loss/q_loss"]:.4f}\n"""
                f"""{'Logq(z|s):':>{pad}} {self.running_logq_zs:.4f}\n"""
                f"""{'Alpha loss:':>{pad}} {loss_dict["Alpha/alpha_loss"]:.4f}\n"""
            )
        if len(rewbuffer) > 0:
            extra_string = (
                f"""{'Mean reward:':>{pad}} {mean_episode_reward:.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(lenbuffer):.2f}\n"""
            )
        log_string = log_string + extra_string

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {total_sample_step}\n"""
            f"""{'Iteration time:':>{pad}} {single_iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.total_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.total_time / (it + 1) * (
                               self.config['max_n_iterations'] - it):.1f}s\n"""
        )
        print(log_string)


    def log_train(self, loss_dict, train_iter):
        for k,v in loss_dict.items():
            self.writer.add_scalar(k, v, train_iter)

    def _save_weights(self, episode, total_sample_step, idx,*rng_states):
        indexed_checkpoint_dir = os.path.join(self.checkpoint_dir, f"checkpoint_{idx}")
        os.mkdir(indexed_checkpoint_dir)
        self.agent.vec_memory.save_buffer(indexed_checkpoint_dir + "/replay.pt")
        torch.save({"policy_network_state_dict": self.agent.policy_network.state_dict(),
                    "q_value_network1_state_dict": self.agent.q_value_network1.state_dict(),
                    "q_value_network2_state_dict": self.agent.q_value_network2.state_dict(),
                    "target_qf1_state_dict": self.agent.target_qf1.state_dict(),
                    "target_qf2_state_dict": self.agent.target_qf2.state_dict(),
                    "discriminator_state_dict": self.agent.discriminator.state_dict(),
                    "q_value1_opt_state_dict": self.agent.q_value1_opt.state_dict(),
                    "q_value2_opt_state_dict": self.agent.q_value2_opt.state_dict(),
                    "policy_opt_state_dict": self.agent.policy_opt.state_dict(),
                    "discriminator_opt_state_dict": self.agent.discriminator_opt.state_dict(),
                    "episode": episode,
                    "total_sample_steps": total_sample_step,
                    "rng_states": rng_states,
                    "max_episode_reward": self.max_episode_reward,
                    "running_logq_zs": self.running_logq_zs
                    } | ({} if not self.agent.auto_entropy_tuning else {
                        "log_alpha": self.agent.log_alpha.item(),
                        "alpha_optimizer_state_dict": self.agent.alpha_optimizer.state_dict(),
                    }),
                   indexed_checkpoint_dir + "/params.pth")

    def load_weights_from_params(self, params_path, load_replay_buffer):
        checkpoint = torch.load(params_path)

        if load_replay_buffer:
            replay_fn = params_path.replace("params.pth", "replay.pt")
            self.agent.vec_memory.load_buffer(replay_fn)

        if self.agent.auto_entropy_tuning:
            self.agent.init_entropy(checkpoint["log_alpha"])
            self.agent.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])
        try:
            self.agent.policy_network.load_state_dict(checkpoint["policy_network_state_dict"])
            self.agent.q_value_network1.load_state_dict(checkpoint["q_value_network1_state_dict"])
            self.agent.q_value_network2.load_state_dict(checkpoint["q_value_network2_state_dict"])
            self.agent.target_qf1.load_state_dict(checkpoint["target_qf1_state_dict"])
            self.agent.target_qf2.load_state_dict(checkpoint["target_qf2_state_dict"])
            self.agent.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
            self.agent.q_value1_opt.load_state_dict(checkpoint["q_value1_opt_state_dict"])
            self.agent.q_value2_opt.load_state_dict(checkpoint["q_value2_opt_state_dict"])
            self.agent.policy_opt.load_state_dict(checkpoint["policy_opt_state_dict"])
            self.agent.discriminator_opt.load_state_dict(checkpoint["discriminator_opt_state_dict"])

            self.max_episode_reward = checkpoint["max_episode_reward"]
            self.running_logq_zs = checkpoint["running_logq_zs"]
        except RuntimeError as e:
            print('Ignoring "' + str(e) + '"')

        return checkpoint["episode"], checkpoint["total_sample_steps"], self.running_logq_zs, checkpoint["rng_states"][0]


    def load_weights(self, model_path, load_replay_buffer=True):
        # model_dir_matchstring = os.getcwd() + "/Checkpoints/" + self.config["env_name"][:-3] + "/" + ((self.config["agent_name"] + "/") if self.config["agent_name"] != "" else "") + "**/params.pth"
        # print(model_dir_matchstring)
        # model_dir = list(glob.glob(model_dir_matchstring))
        # model_dir.sort()
        # print(model_dir)
        # params_path = model_dir[-1]
        return self.load_weights_from_params(model_path,
                                        load_replay_buffer=load_replay_buffer)