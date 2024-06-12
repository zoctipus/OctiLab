from abc import ABC
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
import math
import numpy as np


def init_weight(layer, initializer="he normal"):
    if initializer == "xavier uniform":
        nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.zero_()
    elif initializer == "he normal":
        nn.init.kaiming_normal_(layer.weight)
        layer.bias.data.zero_()
    elif initializer == "gaussian":
        nn.init.normal_(layer.weight, 0, 0.02)
        layer.bias.data.zero_()
    elif initializer == "uniform":
        nn.init.uniform_(layer.weight, -3e-3, 3e-3)
        layer.bias.data.zero_()


class Discriminator(nn.Module, ABC):
    def __init__(self, n_states, n_skills, n_hidden_filters=256):
        super(Discriminator, self).__init__()
        self.n_states = n_states
        self.n_skills = n_skills
        self.n_hidden_filters = n_hidden_filters

        self.hidden1 = nn.Linear(in_features=self.n_states, out_features=self.n_hidden_filters)
        self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        self.q = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_skills)
        self.leakyReLu = nn.LeakyReLU(negative_slope=0.2) # optionally use leakyReLU, see https://machinelearningmastery.com/how-to-train-stable-generative-adversarial-networks/

        self.init_layers()

    def init_layers(self):
        init_weight(self.hidden1, "he normal")
        init_weight(self.hidden2, "he normal")
        init_weight(self.q, "gaussian")

    def forward(self, states):
        x = F.relu(self.hidden1(states))
        x = F.relu(self.hidden2(x))
        logits = self.q(x)
        return logits


class QvalueNetwork(nn.Module, ABC):
    def __init__(self, n_states, n_actions, n_hidden_filters=256):
        super(QvalueNetwork, self).__init__()
        self.n_states = n_states
        self.n_hidden_filters = n_hidden_filters
        self.n_actions = n_actions

        self.hidden1 = nn.Linear(in_features=self.n_states + self.n_actions, out_features=self.n_hidden_filters)
        self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        self.q_value = nn.Linear(in_features=self.n_hidden_filters, out_features=1)
        self.layernorm1 = nn.LayerNorm(self.n_hidden_filters)
        self.layernorm2 = nn.LayerNorm(self.n_hidden_filters)

        #self.init_layers()

    def init_layers(self):
        init_weight(self.q_value, initializer="uniform")

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        x = self.hidden1(x)
        x = self.layernorm1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = self.layernorm2(x)
        x = F.relu(x)
        return self.q_value(x)


class PolicyNetwork(nn.Module, ABC):
    def __init__(self, n_states, n_actions, action_bounds, n_hidden_filters=256):
        super(PolicyNetwork, self).__init__()
        self.n_states = n_states
        self.n_hidden_filters = n_hidden_filters
        self.n_actions = n_actions
        self.set_action_bounds(action_bounds)

        self.hidden1 = nn.Linear(in_features=self.n_states, out_features=self.n_hidden_filters)
        self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        self.mu = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_actions)
        self.log_std = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_actions)

        #self.init_layers()

    def set_action_bounds(self, action_bounds):
        ones = np.ones_like(action_bounds[0])
        ones[action_bounds[0] == action_bounds[1]] = 0.
        assert np.array_equal(action_bounds[0], -ones), "Action Space should be [-1] ~ [1]"
        assert np.array_equal(action_bounds[1], ones), "Action Space should be [-1] ~ [1]"
        # Deprecated
        # - Policy output should be between -1~1 for easier training
        # - Leave it to the environment to scale or clip the action
        self.action_bounds = action_bounds
        action_bounds_low = torch.from_numpy(action_bounds[0])
        action_bounds_high = torch.from_numpy(action_bounds[1])
        self.action_scaler_multiplier = torch.nn.Parameter((action_bounds_high - action_bounds_low) / 2)
        self.action_scaler_add = torch.nn.Parameter((action_bounds_high + action_bounds_low) / 2)
        idx = self.action_scaler_multiplier != 0
        self.action_scaler_log_prob = torch.nn.Parameter(torch.log(self.action_scaler_multiplier[idx]).sum())

    def init_layers(self):
        init_weight(self.mu, initializer="uniform")
        init_weight(self.log_std, initializer="uniform")

    def forward(self, states):
        x = F.relu(self.hidden1(states))
        x = F.relu(self.hidden2(x))
        mu = self.mu(x)
        log_std = self.log_std(x)
        std = log_std.clamp(min=-20, max=2).exp()
        dist = Normal(mu, std)
        return dist

    def get_mean_action(self, states):
        dist = self(states)
        action = torch.tanh(dist.loc)
        return action

    def sample_or_likelihood(self, states):
        dist = self(states)
        # Reparameterization trick
        u = dist.rsample()
        raw_log_prob = dist.log_prob(value=u)
        # Follow rlkit use a more stable formula to compute correction
        # https://github.com/rail-berkeley/rlkit/blob/c81509d982b4d52a6239e7bfe7d2540e3d3cd986/rlkit/torch/distributions.py#L352
        # log(1-tan(u)^2) = 2(log(2)-u-softplus(-2u))
        # softplus is more numerically stable
        correction = 2 * (math.log(2.) - u - F.softplus(-2. * u))
        log_prob = raw_log_prob.sum(-1, keepdim=True) - correction.sum(-1, keepdim=True)
        # Enforcing action bounds
        action = torch.tanh(u) # batch x action
        return action, log_prob
