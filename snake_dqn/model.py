import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, config):
        super().__init__()
        hidden1 = config.get("hidden1", 128)
        hidden2 = config.get("hidden2", 64)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1), nn.ReLU(),
            nn.Linear(hidden1, hidden2), nn.ReLU(),
            nn.Linear(hidden2, output_dim)
        )

    def forward(self, x):
        return self.net(x)