import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from .actuator_networks import ActuatorNetLSTM


# Define the dataset
class ActuatorDataset(Dataset):
    def __init__(self, joint_positions, joint_velocities, torques):
        self.joint_positions = joint_positions
        self.joint_velocities = joint_velocities
        self.torques = torques

    def __len__(self):
        return len(self.joint_positions)

    def __getitem__(self, idx):
        delta_pos = self.joint_positions[idx]
        vel = self.joint_velocities[idx]
        torque = self.torques[idx]
        input_data = np.stack((delta_pos, vel), axis=-1)
        return torch.tensor(input_data, dtype=torch.float32), torch.tensor(torque, dtype=torch.float32)


# Example data
joint_positions = np.random.rand(1000, 1, 1)  # Example joint position differences
joint_velocities = np.random.rand(1000, 1, 1)  # Example joint velocities
torques = np.random.rand(1000, 1)  # Example torques

dataset = ActuatorDataset(joint_positions, joint_velocities, torques)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define the model
model = ActuatorNetLSTM(input_dim=2, hidden_dim=8, num_layers=2, output_dim=1)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(model.linear.weight.device), targets.to(model.linear.weight.device)

        # Initialize hidden states
        hidden_state = (torch.zeros(model.lstm.num_layers, inputs.size(0), model.lstm.hidden_size).to(inputs.device),
                        torch.zeros(model.lstm.num_layers, inputs.size(0), model.lstm.hidden_size).to(inputs.device))

        # Forward pass
        outputs, hidden_state = model(inputs, hidden_state)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), 'actuator_net_lstm.pth')
