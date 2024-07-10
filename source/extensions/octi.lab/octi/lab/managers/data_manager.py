from __future__ import annotations
from collections.abc import Sequence
from prettytable import PrettyTable
from typing import TYPE_CHECKING, Tuple
from omni.isaac.lab.managers.manager_base import ManagerBase, ManagerTermBase
from .manager_term_cfg import DataTermCfg, DataGroupCfg
import torch
if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv


class History:

    def __init__(self, history_length, num_envs, device):
        self.history: dict[str, torch.Tensor] = {}
        self.device = device
        self.begin_idx = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.length = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.history_length = torch.ones(num_envs, dtype=torch.long, device=device) * history_length
        self.num_envs = num_envs
        self.num_envs_index = torch.arange(self.num_envs, device=device)
        self.running_sum = {}  # Stores running sum for each key for each environment
        self.running_sq_sum = {}  # Stores running squared sum for each key for each environment

    def create_history_deque(self, name, value_dimension):
        # Ensure value_dimension is a tuple
        if isinstance(value_dimension, int):
            value_dimension = (value_dimension,)
        dimensions = (self.num_envs, self.history_length[0]) + value_dimension
        self.history[name] = torch.zeros(dimensions, device=self.device)
        self.running_sum[name] = torch.zeros((self.num_envs,) + value_dimension, device=self.device)
        self.running_sq_sum[name] = torch.zeros((self.num_envs,) + value_dimension, device=self.device)

    def update(self, data: dict):
        for key, value in data.items():
            index_tuple = (self.num_envs_index, self.begin_idx)
            # Calculate the indices for the current values to be replaced
            last_values = self.history[key][index_tuple]
            # Update the history buffer
            self.history[key][index_tuple] = value
            # Update running sums
            # This will be correct because when buffer is empty,
            self.running_sum[key] += value - last_values
            self.running_sq_sum[key] += (value**2 - last_values**2)

        self.begin_idx = (self.begin_idx + 1) % self.history_length
        self.length = torch.clamp(self.length + 1, max=self.history_length)

    def reset(self, env_ids: torch.Tensor):
        for key in self.running_sum.keys():
            self.running_sum[key][env_ids] = 0
            self.running_sq_sum[key][env_ids] = 0
            self.history[key][env_ids, :] = 0
        self.begin_idx[env_ids] = 0
        self.length[env_ids] = 0

    def get_data_history(self, name) -> torch.Tensor:
        data = self.history[name]
        ordered_data = []
        for i in range(self.num_envs):
            if self.length[i] < self.history_length:
                ordered_data.append(data[:self.length[i], i][::-1])
            else:
                ordered_data.append(torch.cat((data[self.begin_idx[i]:, i], data[:self.begin_idx[i], i]))[::-1])
        return torch.stack(ordered_data, dim=1)

    def get_mean(self, name: str) -> torch.Tensor:
        # Compute the mean using the running sum and the number of valid entries
        return self.running_sum[name] / self.length.unsqueeze(-1)

    def get_mean_and_std_dev(self, name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute the standard deviation using running sum and squared sum
        mean = self.get_mean(name)
        var = (self.running_sq_sum[name] / self.length.unsqueeze(-1)) - mean**2
        var = torch.clamp(var, min=0)
        return mean, torch.sqrt(var)


class DataManager(ManagerBase):
    """Manager for computing Data signals for a given world.

    Datas are organized into groups based on their intended usage. This allows having different Data
    groups for different types of learning such as asymmetric actor-critic and student-teacher training. Each
    group contains Data terms which contain information about the Data function to call, the noise
    corruption model to use, and the sensor to retrieve data from.

    Each Data group should inherit from the :class:`DataGroupCfg` class. Within each group, each
    Data term should instantiate the :class:`DataTermCfg` class.
    """

    def __init__(self, cfg: object, env: ManagerBasedEnv):
        """Initialize Data manager.

        Args:
            cfg: The configuration object or dictionary (``dict[str, DataGroupCfg]``).
            env: The environment instance.
        """
        super().__init__(cfg, env)

    def __str__(self) -> str:
        """Returns: A string representation for the Data manager."""
        msg = f"<DataManager> contains {len(self._group_obs_term_names)} groups.\n"

        # add info for each group
        for group_name in self._group_obs_term_names.keys():
            # create table for term information
            table = PrettyTable()
            table.title = f"Active Data Terms in Group: '{group_name}'"
            table.field_names = ["Index", "Name", "Shape"]
            # set alignment of table columns
            table.align["Name"] = "l"
            # add info for each term
            obs_terms = zip(
                self._group_obs_term_names[group_name],
                self._group_obs_term_dim[group_name],
            )
            for index, (name, dims) in enumerate(obs_terms):
                # resolve inputs to simplify prints
                tab_dims = tuple(dims)
                # add row
                table.add_row([index, name, tab_dims])
            # convert table to string
            msg += table.get_string()
            msg += "\n"

        return msg

    """
    Properties.
    """

    @property
    def active_terms(self) -> dict[str, list[str]]:
        """Name of active Data terms in each group."""
        return self._group_obs_term_names

    @property
    def group_obs_term_dim(self) -> dict[str, list[tuple[int, ...]]]:
        """Shape of Data tensor for each term in each group."""
        return self._group_obs_term_dim

    @property
    def group_obs_concatenate(self) -> dict[str, bool]:
        """Whether the Data terms are concatenated in each group."""
        return self._group_obs_concatenate

    """
    Operations.
    """
    def get_active_term(self, group_name, term_name) -> torch.Tensor:
        return self._group_data_collections[group_name][term_name]

    def get_history(self, history_group_name) -> History:
        return self._group_data_collections[history_group_name]

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        # call all terms that are classes
        for history_group_name in self.history_group_names:
            self._group_data_collections[history_group_name].reset(env_ids)
        for group_cfg in self._group_obs_class_term_cfgs.values():
            for term_cfg in group_cfg:
                term_cfg.func.reset(env_ids=env_ids)
        # nothing to log here
        return {}

    def compute(self):
        """Compute the Datas per group for all groups.

        The method computes the Datas for all the groups handled by the Data manager.
        Please check the :meth:`compute_group` on the processing of Datas per group.

        Returns:
            A dictionary with keys as the group names and values as the computed Datas.
        """
        # create a buffer for storing obs from all the groups
        obs_buffer = dict()
        # iterate over all the terms in each group
        for group_name in self._group_obs_term_names:
            obs_buffer[group_name] = self.compute_group(group_name)
            self._group_data_collections[group_name].update(obs_buffer[group_name])

    def compute_group(self, group_name: str) -> torch.Tensor | dict[str, torch.Tensor]:
        """Computes the Datas for a given group.

        The Datas for a given group are computed by calling the registered functions for each
        term in the group. The functions are called in the order of the terms in the group. The functions
        are expected to return a tensor with shape (num_envs, ...).

        If a corruption/noise model is registered for a term, the function is called to corrupt
        the Data. The corruption function is expected to return a tensor with the same
        shape as the Data. The Datas are clipped and scaled as per the configuration
        settings.

        The operations are performed in the order: compute, add corruption/noise, clip, scale.
        By default, no scaling or clipping is applied.

        Args:
            group_name: The name of the group for which to compute the Datas. Defaults to None,
                in which case Datas for all the groups are computed and returned.

        Returns:
            Depending on the group's configuration, the tensors for individual Data terms are
            concatenated along the last dimension into a single tensor. Otherwise, they are returned as
            a dictionary with keys corresponding to the term's name.

        Raises:
            ValueError: If input ``group_name`` is not a valid group handled by the manager.
        """
        # check ig group name is valid
        if group_name not in self._group_obs_term_names:
            raise ValueError(
                f"Unable to find the group '{group_name}' in the Data manager."
                f" Available groups are: {list(self._group_obs_term_names.keys())}"
            )
        # iterate over all the terms in each group
        group_term_names = self._group_obs_term_names[group_name]
        # buffer to store obs per group
        group_obs = dict.fromkeys(group_term_names, None)
        # # read attributes for each term
        # obs_terms = zip(group_term_names, self._group_obs_term_cfgs[group_name])
        # evaluate terms: compute, add noise, clip, scale.
        # for name, term_cfg in obs_terms:
        # compute term's value
        for term_cfg in self._group_obs_term_cfgs[group_name]:
            data_dict: dict[str, torch.Tensor] = term_cfg.func(self._env, **term_cfg.params)
            for key, value in data_dict.items():
                # apply post-processing
                if term_cfg.noise:
                    value = term_cfg.noise.func(value, term_cfg.noise)
                if term_cfg.clip:
                    value = value.clip_(min=term_cfg.clip[0], max=term_cfg.clip[1])
                if term_cfg.scale:
                    value = value.mul_(term_cfg.scale)
                # TODO: Introduce delay and filtering models.
                # Ref: https://robosuite.ai/docs/modules/sensors.html#observables
                # add value to list
                group_obs[key] = value
        # concatenate all Datas in the group together
        if self._group_obs_concatenate[group_name]:
            return torch.cat(list(group_obs.values()), dim=-1)
        else:
            return group_obs

    """
    Helper functions.
    """

    def _prepare_terms(self):
        """Prepares a list of Data terms functions."""
        # create buffers to store information for each Data group
        # TODO: Make this more convenient by using data structures.
        self._group_data_collections: dict[str, dict[str, torch.Tensor] | History] = dict()
        self._group_obs_term_names: dict[str, list[str]] = dict()
        self._group_obs_term_dim: dict[str, list[int]] = dict()
        self._group_obs_term_cfgs: dict[str, list[DataTermCfg]] = dict()
        self._group_obs_class_term_cfgs: dict[str, list[DataTermCfg]] = dict()
        self._group_obs_concatenate: dict[str, bool] = dict()
        self.history_group_names = list()

        # check if config is dict already
        if isinstance(self.cfg, dict):
            group_cfg_items = self.cfg.items()
        else:
            group_cfg_items = self.cfg.__dict__.items()
        # iterate over all the groups
        for group_name, group_cfg in group_cfg_items:
            # check for non config
            if group_cfg is None:
                continue
            # check if the term is a curriculum term
            if not isinstance(group_cfg, DataGroupCfg):
                raise TypeError(
                    f"Data group '{group_name}' is not of type 'DataGroupCfg'."
                    f" Received: '{type(group_cfg)}'."
                )
            # initialize list for the group settings
            self._group_data_collections[group_name] = dict()
            self._group_obs_term_names[group_name] = list()
            self._group_obs_term_dim[group_name] = list()
            self._group_obs_term_cfgs[group_name] = list()
            self._group_obs_class_term_cfgs[group_name] = list()
            # read common config for the group
            self._group_obs_concatenate[group_name] = group_cfg.concatenate_terms
            # check if config is dict already
            if isinstance(group_cfg, dict):
                group_cfg_items = group_cfg.items()
            else:
                group_cfg_items = group_cfg.__dict__.items()
            # iterate over all the terms in each group
            for term_name, term_cfg in group_cfg.__dict__.items():
                # skip non-obs settings
                if term_name in ["enable_corruption", "concatenate_terms"]:
                    continue
                # check for non config
                if term_cfg is None:
                    continue
                if not isinstance(term_cfg, DataTermCfg):
                    raise TypeError(
                        f"Configuration for the term '{term_name}' is not of type DataTermCfg."
                        f" Received: '{type(term_cfg)}'."
                    )
                # resolve common terms in the config
                self._resolve_common_term_cfg(f"{group_name}/{term_name}", term_cfg, min_argc=1)
                # check noise settings
                if not group_cfg.enable_corruption:
                    term_cfg.noise = None
                # add term config to list to list
                self._group_obs_term_names[group_name].append(term_name)
                self._group_obs_term_cfgs[group_name].append(term_cfg)
                # call function the first time to fill up dimensions
                result = term_cfg.func(self._env, **term_cfg.params)
                num_envs = next(iter(result.values())).shape[0]
                if term_cfg.history_length > 1:
                    self.history_group_names.append(group_name)
                    self._group_data_collections[group_name] = History(
                        term_cfg.history_length, num_envs, device=self.device)
                if isinstance(result, dict):
                    self._group_obs_term_names[group_name].remove(term_name)
                    for key, value in result.items():
                        self._group_obs_term_names[group_name].append(key)
                        if isinstance(value, tuple):
                            print("stop")
                        obs_dims = tuple(value.shape[1:])
                        group = self._group_data_collections[group_name]
                        if isinstance(self._group_data_collections[group_name], History):
                            self._group_obs_term_dim[group_name].append((term_cfg.history_length,) + obs_dims)
                            group.create_history_deque(key, obs_dims)
                        else:
                            self._group_obs_term_dim[group_name].append(obs_dims)
                            group[key] = value
                else:
                    raise ValueError(f"Returned Data must be dict[str, Tensor],"
                                     f"type{type(result)} is not allowed ")
                # add term in a separate list if term is a class
                if isinstance(term_cfg.func, ManagerTermBase):
                    self._group_obs_class_term_cfgs[group_name].append(term_cfg)
                    # call reset (in-case above call to get obs dims changed the state)
                    term_cfg.func.reset()
