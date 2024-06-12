import torch
import time
import h5py

class RolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.next_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.zs = None
            

        def clear(self):
            self.__init__()

    def __init__(self, num_envs, max_num_transitions_per_env, obs_shape, actions_shape, device="cpu"):
        self.device = device

        self.obs_shape = obs_shape
        self.actions_shape = actions_shape
        self.time_spent_flattening = 0
        self.time_spent_moving = 0
        self.time_spent_permutation = 0

        self.observations = torch.zeros(max_num_transitions_per_env, num_envs, obs_shape, device=self.device)
        self.next_observations = torch.zeros(max_num_transitions_per_env, num_envs, obs_shape, device=self.device)
        self.actions = torch.zeros(max_num_transitions_per_env, num_envs, actions_shape, device=self.device)
        self.rewards = torch.zeros(max_num_transitions_per_env, num_envs, 1, device=self.device)
        self.dones = torch.zeros(max_num_transitions_per_env, num_envs, 1, device=self.device, dtype=torch.bool)
        self.zs = torch.zeros(max_num_transitions_per_env, num_envs, 1, device=self.device, dtype=torch.long)

        self.max_num_transitions_per_env = max_num_transitions_per_env
        self.num_envs = num_envs

        self.length = 0
        self.step = 0
        self.demo_reserved_idx = 0

    def __len__(self):
        return self.length * self.num_envs
    
    def usage_percentage(self):
        return self.length / self.max_num_transitions_per_env

    def add_transitions(self, transition: Transition):
        # Copy the new transition data into the current position indicated by self.step
        self.observations[self.step].copy_(transition.observations)
        self.next_observations[self.step].copy_(transition.next_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.zs[self.step].copy_(transition.zs.view(-1, 1))

        # Increment the step to point to the next position in the buffer
        self.step += 1

        # If the step reaches the max_num_transitions_per_env, it wraps around to 0
        if self.step >= self.max_num_transitions_per_env:
            self.step = self.demo_reserved_idx

        # Update the length to be the maximum of the current length or the step if it is increasing
        # This ensures that length reaches max_num_transitions_per_env and then stays there
        if self.length < self.max_num_transitions_per_env:
            self.length += 1

    def clear(self):
        self.step = 0
        self.length = 0

    def mini_batch_generator(self, mini_batch_size, num_mini_batches=8):
        batch_size = self.num_envs * self.length
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(self.length * self.num_envs, requires_grad=False, device=self.device)[:batch_size]

        # Flatten the data tensors to enable easy indexing
        observations = self.observations.flatten(0, 1)
        next_observations = self.next_observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        zs = self.zs.flatten(0, 1)
        rewards = self.rewards.flatten(0, 1)
        dones = self.dones.flatten(0, 1)

        for i in range(num_mini_batches):
            start = i * mini_batch_size
            end = (i + 1) * mini_batch_size
            batch_idx = indices[start:end]

            obs_batch = observations[batch_idx]
            next_observations_batch = next_observations[batch_idx]
            actions_batch = actions[batch_idx]
            zs = zs[batch_idx]
            rewards = rewards[batch_idx]
            dones = dones[batch_idx]
            yield obs_batch, next_observations_batch, actions_batch, zs, rewards, dones

    def sample(self, batch_size, device):
        # Calculate the total number of transitions available across all environments
        permutation_start = time.time()
        memory_total_transitions = self.num_envs * self.length
        indices = torch.randint(0, memory_total_transitions, (batch_size,), device=self.device)
        permutation_end = time.time()
        # Flatten the data tensors to enable easy indexing
        flatten_start = time.time()
        observations = self.observations.flatten(0, 1)
        next_observations = self.next_observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        zs = self.zs.flatten(0, 1)
        rewards = self.rewards.flatten(0, 1)
        dones = self.dones.flatten(0, 1)
        flatten_end = time.time()
        move_to_gpu_start = time.time()
        obs_batch = observations[indices].to(device)
        next_observations_batch = next_observations[indices].to(device)
        actions_batch = actions[indices].to(device)
        zs = zs[indices].to(device)
        rewards = rewards[indices].to(device)
        dones = dones[indices].to(device)
        move_to_gpu_end = time.time()
        self.time_spent_moving = move_to_gpu_end - move_to_gpu_start
        self.time_spent_flattening= flatten_end - flatten_start
        self.time_spent_permutation = permutation_end - permutation_start
        # Return the batch
        return obs_batch, next_observations_batch, actions_batch, rewards, dones, zs
    
    def save_buffer(self, path):
        data = {
            'observations': self.observations,
            'next_observations': self.next_observations,
            'actions': self.actions,
            'rewards': self.rewards,
            'dones': self.dones,
            'zs': self.zs,
            'max_num_transitions_per_env': self.max_num_transitions_per_env,
            'num_envs': self.num_envs,
            'step': self.step,
            'length': self.length,
            'demo_reserved_idx': self.demo_reserved_idx
        }
        torch.save(data, path)

    def load_buffer(self, path):
        data = torch.load(path)
        self.observations = data['observations']
        self.next_observations = data['next_observations']
        self.actions = data['actions']
        self.rewards = data['rewards']
        self.dones = data['dones']
        self.zs = data['zs']
        self.max_num_transitions_per_env = data['max_num_transitions_per_env']
        self.num_envs = data['num_envs']
        self.step = data['step']
        self.length = data['length']
        self.demo_reserved_idx = data['demo_reserved_idx']

    def load_buffer_from_hdf5(self, path):
        with h5py.File(path, "r") as f:
            # for demo_key in f["data"].items():
            transition_empty = False
            extra_length = 0

            obs_buffer = torch.zeros(self.num_envs, self.obs_shape, device=self.device)
            next_obs_buffer = torch.zeros(self.num_envs, self.obs_shape, device=self.device)
            actions_buffer = torch.zeros(self.num_envs, self.actions_shape, device=self.device)
            rewards_buffer = torch.zeros(self.num_envs, 1, device=self.device)
            dones_buffer = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.bool)
            zs_buffer = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.long)
            demokeys = [demo_key.decode() for demo_key in f["mask/successful"]]
            buffer_idx = 0
            extra_length = 0
            for demo_key_string in demokeys:
                demo_idx = 0
                observation_data = f[f"data/{demo_key_string}/obs"]
                key_orders = ['robot_eepose', 'canberry_pos_b', 'cake_pos_b']
                obs = {key : torch.from_numpy(f[f"data/{demo_key_string}/obs/{key}"][:]).to(self.device) for key in observation_data}
                next_obs = {key : torch.from_numpy(f[f"data/{demo_key_string}/next_obs/{key}"][:]).to(self.device) for key in observation_data}
                actions = torch.from_numpy(f[f"data/{demo_key_string}/actions"][:]).to(self.device)
                rewards = torch.from_numpy(f[f"data/{demo_key_string}/rewards"][:]).to(self.device)
                dones = torch.from_numpy(f[f"data/{demo_key_string}/dones"][:]).to(self.device)

                extra_length += len(actions)
                
                while extra_length > 0:
                    while buffer_idx < self.num_envs and extra_length > 0:
                        obs_buffer[buffer_idx, :] = torch.cat([obs[key][demo_idx] for key in key_orders] + [torch.tensor([0], device=self.device)], dim=-1)
                        next_obs_buffer[buffer_idx, :] = torch.cat([next_obs[key][demo_idx] for key in key_orders] + [torch.tensor([0], device=self.device)], dim=-1)
                        actions_buffer[buffer_idx, :] = actions[demo_idx].view(1, -1)
                        rewards_buffer[buffer_idx, :] = rewards[demo_idx].view(1, -1)
                        dones_buffer[buffer_idx, :] = dones[demo_idx].view(1, -1)
                        buffer_idx += 1
                        demo_idx += 1
                        extra_length -= 1
                    
                    if buffer_idx == self.num_envs:
                        transition = self.Transition()
                        transition.observations = obs_buffer.clone()
                        transition.next_observations = next_obs_buffer.clone()
                        transition.actions = actions_buffer.clone()
                        transition.rewards = rewards_buffer.clone()
                        transition.dones = dones_buffer.clone()
                        transition.zs = zs_buffer.clone()

                        self.add_transitions(transition)

                        self.demo_reserved_idx += 1
                        buffer_idx = 0 
                
                if self.demo_reserved_idx >= self.max_num_transitions_per_env * 0.5:
                    break
        
        print(f"done adding transitions {self.demo_reserved_idx * self.num_envs} transitions")


    @staticmethod
    def get_rng_state():
        return torch.get_rng_state()

    @staticmethod
    def set_rng_state(random_rng_state):
        torch.set_rng_state(random_rng_state)