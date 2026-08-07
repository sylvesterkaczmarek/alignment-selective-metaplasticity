from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, width),
        )
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + 0.5 * self.net(x))


class SelectiveCorrigibilityNet(nn.Module):
    """Small shared network used for all proxy and capability tasks."""

    def __init__(self, input_dim: int, hidden_dim: int = 48, depth: int = 2) -> None:
        super().__init__()
        self.input = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(depth)])
        self.output = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input(x)
        h = self.blocks(h)
        return self.output(h)
