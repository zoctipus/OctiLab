import torch
import torch.nn as nn


class ActuatorNetLSTM(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=8, num_layers=2, output_dim=1):
        super(ActuatorNetLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden_state=None):
        if hidden_state is None:
            # Initialize hidden states if not provided
            hidden_state = (torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device),
                            torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device))

        lstm_out, hidden_state = self.lstm(x, hidden_state)
        output = self.linear(lstm_out[:, -1, :])
        return output, hidden_state


class ActuatorNetwork(nn.Module):
    # 3 position errors + 3 velocities + 1 motor type
    def __init__(self, input_dim=7, hidden_dim=32, output_dim=1):
        super(ActuatorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Softsign()  # or nn.Tanh()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)  # No activation here for regression
        return x
