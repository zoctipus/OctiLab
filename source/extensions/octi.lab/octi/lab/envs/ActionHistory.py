import torch


class ActionHistory:
    def __init__(self, action_capacity, num_env, action_space_size):
        self.action_capacity = action_capacity
        self.num_env = num_env
        self.action_space_size = action_space_size
        # Actions stored per environment
        self.actions = torch.zeros((num_env, action_capacity, action_space_size))
        # Current index for each environment
        self.current_index = torch.zeros(num_env)
        # Track the number of actions added per environment
        self.size = torch.zeros(num_env)

    def add_action(self, env_idx, action):
        # Add action for a specific environment
        idx = self.current_index[env_idx]
        self.actions[env_idx, idx] = action
        self.current_index[env_idx] = (idx + 1) % self.action_capacity
        self.size[env_idx] = min(self.size[env_idx] + 1, self.action_capacity)

    def get_average(self, env_idx, number):
        # Assuming env_idx is an index for a single environment, not a tensor of indices
        num_actions = torch.min(self.size[env_idx], torch.tensor(number)).item()
        if num_actions == 0:
            return torch.zeros((self.action_space_size,))

        start_idx = (self.current_index[env_idx] - num_actions) % self.action_capacity
        end_idx = self.current_index[env_idx]

        if start_idx < end_idx:
            action_subset = self.actions[env_idx, start_idx:end_idx]
        else:
            action_subset = torch.cat((self.actions[env_idx, start_idx:], self.actions[env_idx, :end_idx]), dim=0)

        latest_actions_sum = action_subset.sum(dim=0)
        average = latest_actions_sum / num_actions
        return average

    def reset(self, reset_idx):
        # Reset specific environments indicated by reset_idx
        for idx in reset_idx:
            self.actions[idx] = torch.zeros((self.action_capacity, self.action_space_size))
            self.current_index[idx] = 0
            self.size[idx] = 0


def test_action_history():
    # Configuration for the test
    action_capacity = 100
    num_env = 1
    action_space_size = 3
    num_actions = 1000

    # Initialize ActionHistory and a list to store actions
    action_history = ActionHistory(action_capacity, num_env, action_space_size)
    action_list = []

    # Generate and add actions
    for _ in range(num_actions):
        action = torch.rand((num_env, action_space_size))  # Generate a random action
        env_idx = 0  # Assuming a single environment
        action_history.add_action(env_idx, action)
        action_list.append(action)

    # Ensure only the last 'action_capacity' actions are stored
    expected_actions = action_list[-action_capacity:]

    # Test get method
    retrieved_actions = action_history.get(env_idx, action_capacity)
    assert torch.allclose(retrieved_actions, torch.stack(expected_actions)), "get method failed"

    # Test get_average method for different numbers of actions
    for num in [10, 50, 100, 200]:
        expected_average = torch.mean(torch.stack(expected_actions[-num:]), dim=0)
        average_actions = action_history.get_average(env_idx, num)
        assert torch.allclose(average_actions, expected_average), f"get_average method failed for num={num}"

    print("All tests passed!")
