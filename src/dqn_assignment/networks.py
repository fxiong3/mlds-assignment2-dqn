"""Q-network architectures for the two assignment environments."""

from __future__ import annotations

import torch
from torch import nn


class MLPQNetwork(nn.Module):
    """Compact multilayer perceptron for CartPole state vectors."""

    def __init__(self, input_dim: int, action_count: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_count),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.layers(observations)


class AtariQNetwork(nn.Module):
    """DQN convolutional network adapted to 88x80 grayscale observations."""

    def __init__(self, input_channels: int, action_count: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        with torch.no_grad():
            feature_count = self.features(torch.zeros(1, input_channels, 88, 80)).numel()
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_count, 512),
            nn.ReLU(),
            nn.Linear(512, action_count),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(observations))
