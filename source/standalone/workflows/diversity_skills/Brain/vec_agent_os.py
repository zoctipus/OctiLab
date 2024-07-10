import numpy as np
from .model import PolicyNetwork, QvalueNetwork, Discriminator
import torch
from .memory import RolloutStorage
from torch import from_numpy
from torch.optim.adam import Adam
from torch.nn.functional import log_softmax
from functools import reduce
import time
import operator


class DSACAgent:
    def __init__(self,
                 p_z,
                 config,
                 replay_buffer_device,
                 replay_buffer_loading_path = None):
        self.config = config
        self.output_scale = 0.1
        #TODO:currently only 1d state 1d action is supported
        self.n_states = self.config["n_states"][0]
        self.n_actions = self.config["n_actions"][0]
        self.n_envs = self.config["n_envs"]
        self.n_skills = self.config["n_skills"]
        self.batch_size = self.config["batch_size"]
        self.p_z = np.tile(p_z, self.batch_size).reshape(self.batch_size, self.n_skills)
        self.transition = RolloutStorage.Transition()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vec_memory = RolloutStorage(self.n_envs, self.config["mem_size"], self.n_states + 1, self.n_actions, device=replay_buffer_device)

        self.policy_network = PolicyNetwork(n_states=self.n_states + self.n_skills,
                                            n_actions=self.n_actions,
                                            action_bounds=self.config["action_bounds"],
                                            n_hidden_filters=self.config["n_hiddens"]).to(self.device)

        self.q_value_network1 = QvalueNetwork(n_states=self.n_states + self.n_skills,
                                              n_actions=self.n_actions,
                                              n_hidden_filters=self.config["n_hiddens"]).to(self.device)

        self.q_value_network2 = QvalueNetwork(n_states=self.n_states + self.n_skills,
                                              n_actions=self.n_actions,
                                              n_hidden_filters=self.config["n_hiddens"]).to(self.device)

        self.target_qf1 = QvalueNetwork(n_states=self.n_states + self.n_skills,
                                        n_actions=self.n_actions,
                                        n_hidden_filters=self.config["n_hiddens"]).to(self.device)

        self.target_qf2 = QvalueNetwork(n_states=self.n_states + self.n_skills,
                                        n_actions=self.n_actions,
                                        n_hidden_filters=self.config["n_hiddens"]).to(self.device)
        self.hard_update_target_network(self.q_value_network1, self.target_qf1)
        self.hard_update_target_network(self.q_value_network2, self.target_qf2)

        self.discriminator = Discriminator(n_states=self.n_states, n_skills=self.n_skills,
                                           n_hidden_filters=self.config["n_hiddens"]).to(self.device)

        self.mse_loss = torch.nn.MSELoss()
        self.cross_ent_loss = torch.nn.CrossEntropyLoss()

        self.q_value1_opt = Adam(self.q_value_network1.parameters(), lr=self.config["lr"])
        self.q_value2_opt = Adam(self.q_value_network2.parameters(), lr=self.config["lr"])
        self.policy_opt = Adam(self.policy_network.parameters(), lr=self.config["lr"])
        self.discriminator_opt = Adam(self.discriminator.parameters(), lr=self.config["lr"])

        # Tunable Entropy
        self.auto_entropy_tuning = config["auto_entropy_tuning"]
        if self.auto_entropy_tuning:
            self.init_entropy()
            print("Enable auto entropy tuning")

        if replay_buffer_loading_path is not None:
            self.vec_memory.load_buffer_from_hdf5(replay_buffer_loading_path)

    def init_entropy(self, value=0.):
        self.target_entropy = (
            self.config["alpha"] or -self.n_actions)  # heuristic target entropy
        print("Target entropy H (log) is ", self.target_entropy)
        self.target_entropy = torch.tensor(self.target_entropy, requires_grad=False, device=self.device)
        self.log_alpha = torch.tensor(value, requires_grad=True, device=self.device)
        self.alpha_optimizer = Adam(
            [self.log_alpha],
            lr=self.config["lr"],
        )

    def choose_deterministic_action(self, states):
        states = torch.unsqueeze(states, dim=0)
        with torch.no_grad():
            action = self.policy_network.get_mean_action(states)
        return action[0] * self.output_scale
    
    def vec_choose_action(self, states):
        # states = from_numpy(states).float().to(self.device)
        with torch.no_grad():
            action, _ = self.policy_network.sample_or_likelihood(states)
        return action.detach() * self.output_scale

    def vec_store(self, state, z, done, action, next_state, reward):
        self.transition.observations = state
        self.transition.next_observations = next_state
        self.transition.actions = action
        self.transition.rewards = reward
        self.transition.dones = done
        self.transition.zs = z
        self.vec_memory.add_transitions(self.transition)

    def discriminator_reward(self, state, z):
        with torch.no_grad():
            states = np.expand_dims(state, axis=0)
            states = from_numpy(states).float().to(self.device)
            p_z = from_numpy(self.p_z[0:1,:]).to(self.device)
            zs = from_numpy(np.array([[z]])).to(self.device)
            logits = self.discriminator(torch.split(states, [self.n_states, self.n_skills], dim=-1)[0])
            logq_z_ns = log_softmax(logits, dim=-1)
            logq_z_s = logq_z_ns.gather(-1, zs).detach()
            d_rew = (logq_z_s - torch.log(p_z.gather(-1, zs) + 1e-6)).float()
            return torch.exp(logq_z_s)[-1].item(), d_rew[-1].item()

    def train(self, diversity_reward=True) -> dict:
        replay_buffer_sample_time_per_iteration = 0
        train_time_per_iteration = 0
        if len(self.vec_memory) < self.batch_size:
            return None
        else:
            start_sampling = time.time()
            states,next_states,actions,env_rewards,dones,zs = self.vec_memory.sample(self.batch_size, self.device)
            p_z = from_numpy(self.p_z).to(self.device)
            end_sampling = time.time()
            replay_buffer_sample_time_per_iteration += end_sampling - start_sampling

            """
            Alpha Loss
            """
            start_training_time = time.time()
            new_obs_actions, log_probs = self.policy_network.sample_or_likelihood(states)
            if self.auto_entropy_tuning:
                alpha_loss = -(self.log_alpha.exp() * (log_probs + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                alpha = self.log_alpha.exp().detach().item()
            else:
                alpha_loss = 0
                alpha = self.config["alpha"]

            """
            Policy Loss
            """
            q_new_actions = torch.min(
                self.q_value_network1(states, new_obs_actions),
                self.q_value_network2(states, new_obs_actions),
            )

            # Standard SAC (with reparam trick a \sim policy): J_pi = log pi(a|s) - q(s,a)
            policy_loss = (alpha * log_probs - q_new_actions).mean()
            self.policy_opt.zero_grad()
            policy_loss.backward()
            self.policy_opt.step()

            """
            Rewards
            """
            rewards = env_rewards if not self.config["omit_env_rewards"] else 0

            # DIAYN: define r = H(A|S,Z) + E_{z \sim p(z), s \sim pi(z)}[log discriminator(z|s) - log p(z)]
            # TODO: what what to do with this diversity_reward
            # if diversity_reward:
            if False:
                with torch.no_grad():
                    next_logits = self.discriminator(torch.split(next_states, [self.n_states, self.n_skills], dim=-1)[0])
                    logq_z_ns = log_softmax(next_logits, dim=-1)
                    disciminator_reward = (logq_z_ns.gather(-1, zs).detach() - torch.log(p_z.gather(-1, zs) + 1e-6)).float()
                rewards += self.config["reward_balance"] * disciminator_reward

            """
            Q Function Loss
            """
            with torch.no_grad():
                reparam_next_actions, next_log_probs = self.policy_network.sample_or_likelihood(next_states)
                target_q_values = torch.min(
                    self.target_qf1(next_states, reparam_next_actions),
                    self.target_qf2(next_states, reparam_next_actions),
                ) - alpha * next_log_probs.detach()

                # Standard SAC: J_Q = Q(s,a) - (r + gamma * min_Q(s+1,a+1))
                target_q = self.config["reward_scale"] * rewards.float() + \
                           self.config["gamma"] * target_q_values * (~dones)

            q1_pred = self.q_value_network1(states, actions)
            q2_pred = self.q_value_network2(states, actions)
            q1_loss = self.mse_loss(q1_pred, target_q)
            q2_loss = self.mse_loss(q2_pred, target_q)

            self.q_value1_opt.zero_grad()
            q1_loss.backward()
            self.q_value1_opt.step()

            self.q_value2_opt.zero_grad()
            q2_loss.backward()
            self.q_value2_opt.step()

            """
            Discriminator loss
            """
            logits = self.discriminator(torch.split(states, [self.n_states, self.n_skills], dim=-1)[0])
            discriminator_loss = self.cross_ent_loss(logits, zs.squeeze(-1))
            self.discriminator_opt.zero_grad()
            discriminator_loss.backward()
            self.discriminator_opt.step()

            """
            Update target networks
            """
            self.soft_update_target_network(self.q_value_network1, self.target_qf1)
            self.soft_update_target_network(self.q_value_network2, self.target_qf2)

            end_training_time = time.time()
            train_time_per_iteration += end_training_time - start_training_time

            return {
                'Loss/policy_loss':policy_loss.item(),
                'Loss/q_loss': 0.5 * (q1_loss.item() + q2_loss.item()),
                'Loss/q1_loss':q1_loss.item(),
                'Loss/q2_loss':q2_loss.item(),
                'Loss/discriminator_loss':discriminator_loss.item(),
                'Alpha/log_probs': log_probs.mean().item(),
                'Alpha/alpha': alpha,
                'Alpha/log_alpha': self.log_alpha.item() if self.auto_entropy_tuning else 0,
                'Alpha/alpha_loss': alpha_loss.item() if self.auto_entropy_tuning else alpha_loss,
                'train_sampling_time': replay_buffer_sample_time_per_iteration,
                'train_training_time': train_time_per_iteration
                }

    def train_critic(self, diversity_reward=True)->dict:
        critic_replay_buffer_sample_time_per_iteration = 0
        critic_train_time_per_iteration = 0
        if len(self.vec_memory) < self.batch_size:
            return None
        else:
            start_sampling = time.time()
            states,next_states,actions,env_rewards,dones,zs = self.vec_memory.sample(self.batch_size, self.device)
            end_sampling = time.time()
            start_training_time = time.time()
            p_z = from_numpy(self.p_z).to(self.device)
            alpha = self.log_alpha.exp().detach().item()

            """
            Rewards
            """
            rewards = env_rewards if not self.config["omit_env_rewards"] else 0

            # DIAYN: define r = H(A|S,Z) + E_{z \sim p(z), s \sim pi(z)}[log discriminator(z|s) - log p(z)]
            # TODO: check what to do with diversity_reward
            # if diversity_reward:
            if False:
                with torch.no_grad():
                    next_logits = self.discriminator(torch.split(next_states, [self.n_states, self.n_skills], dim=-1)[0])
                    logq_z_ns = log_softmax(next_logits, dim=-1)
                    disciminator_reward = (logq_z_ns.gather(-1, zs).detach() - torch.log(p_z.gather(-1, zs) + 1e-6)).float()
                rewards += self.config["reward_balance"] * disciminator_reward

            """
            Q Function Loss
            """
            with torch.no_grad():
                reparam_next_actions, next_log_probs = self.policy_network.sample_or_likelihood(next_states)
                target_q_values = torch.min(
                    self.target_qf1(next_states, reparam_next_actions),
                    self.target_qf2(next_states, reparam_next_actions),
                ) - alpha * next_log_probs.detach()

                # Standard SAC: J_Q = Q(s,a) - (r + gamma * min_Q(s+1,a+1))
                target_q = self.config["reward_scale"] * rewards.float() + \
                           self.config["gamma"] * target_q_values * (~dones)

            q1_pred = self.q_value_network1(states, actions)
            q2_pred = self.q_value_network2(states, actions)
            q1_loss = self.mse_loss(q1_pred, target_q)
            q2_loss = self.mse_loss(q2_pred, target_q)

            self.q_value1_opt.zero_grad()
            q1_loss.backward()
            self.q_value1_opt.step()

            self.q_value2_opt.zero_grad()
            q2_loss.backward()
            self.q_value2_opt.step()

            """
            Discriminator loss
            """
            logits = self.discriminator(torch.split(states, [self.n_states, self.n_skills], dim=-1)[0])
            discriminator_loss = self.cross_ent_loss(logits, zs.squeeze(-1))
            self.discriminator_opt.zero_grad()
            discriminator_loss.backward()
            self.discriminator_opt.step()

            """
            Update target networks
            """
            self.soft_update_target_network(self.q_value_network1, self.target_qf1)
            self.soft_update_target_network(self.q_value_network2, self.target_qf2)
            end_training_time = time.time()

            critic_replay_buffer_sample_time_per_iteration = end_sampling - start_sampling
            critic_train_time_per_iteration = end_training_time - start_training_time
            return {
                'Loss/q_loss': 0.5 * (q1_loss.item() + q2_loss.item()),
                'Loss/q1_loss':q1_loss.item(),
                'Loss/q2_loss':q2_loss.item(),
                'Loss/discriminator_loss':discriminator_loss.item(),
                'critic_sampling_time': critic_replay_buffer_sample_time_per_iteration,
                'critic_training_time': critic_train_time_per_iteration
                }

    def soft_update_target_network(self, local_network, target_network):
        for target_param, local_param in zip(target_network.parameters(), local_network.parameters()):
            target_param.data.copy_(self.config["tau"] * local_param.data +
                                    (1 - self.config["tau"]) * target_param.data)

    def hard_update_target_network(self, local_network, target_network):
        target_network.load_state_dict(local_network.state_dict())
        target_network.eval()

    def hard_update_all_target_networks(self):
        self.hard_update_target_network(self.q_value_network2, self.target_qf2)
        self.hard_update_target_network(self.q_value_network1, self.target_qf1)

    def get_rng_states(self):
        return torch.get_rng_state(), self.vec_memory.get_rng_state()

    def set_rng_states(self, torch_rng_state, random_rng_state):
        torch.set_rng_state(torch_rng_state)
        self.vec_memory.set_rng_state(random_rng_state)

    def set_policy_net_to_eval_mode(self):
        self.policy_network.eval()

    def set_policy_net_to_cpu_mode(self):
        self.device = torch.device("cpu")
        self.policy_network.to(self.device)
        self.discriminator.to(self.device)
