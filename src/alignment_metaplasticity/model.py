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


class SelectiveCorrigibilityTransformer(nn.Module):
    """Small scalar-feature token encoder for an architecture-transfer check."""
    def __init__(self, input_dim: int, hidden_dim: int = 48, depth: int = 2):
        super().__init__()
        if min(input_dim, hidden_dim, depth) < 1 or hidden_dim % 4:
            raise ValueError("positive dimensions and hidden width divisible by four are required")
        self.feature_vectors = nn.Parameter(torch.empty(input_dim, hidden_dim))
        self.feature_bias = nn.Parameter(torch.empty(input_dim, hidden_dim))
        nn.init.normal_(self.feature_vectors, std=.2)
        nn.init.normal_(self.feature_bias, std=.2)
        layer = nn.TransformerEncoderLayer(hidden_dim, 4, dim_feedforward=2*hidden_dim,
                                           dropout=0., batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.output = nn.Linear(input_dim * hidden_dim, 2)

    def forward(self, x):
        tokens = x.unsqueeze(-1) * self.feature_vectors + self.feature_bias
        return self.output(self.encoder(tokens).flatten(1))
