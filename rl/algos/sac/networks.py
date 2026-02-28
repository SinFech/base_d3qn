from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
from torch.distributions import Normal


def _build_mlp(
    input_dim: int,
    hidden_sizes: Iterable[int],
    output_dim: int,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = input_dim
    for size in hidden_sizes:
        layers.append(nn.Linear(current_dim, int(size)))
        layers.append(nn.ReLU())
        current_dim = int(size)
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


class SACGaussianPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: list[int],
        action_low: float,
        action_high: float,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.net = _build_mlp(self.obs_dim, hidden_sizes, self.action_dim * 2)
        action_low_t = torch.full((self.action_dim,), float(action_low), dtype=torch.float32)
        action_high_t = torch.full((self.action_dim,), float(action_high), dtype=torch.float32)
        self.register_buffer("action_low", action_low_t)
        self.register_buffer("action_high", action_high_t)

    def _forward_stats(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        obs = obs.view(obs.size(0), -1)
        out = self.net(obs)
        mean, log_std = torch.chunk(out, 2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def _scale_action(self, squashed_action: torch.Tensor) -> torch.Tensor:
        return self.action_low + 0.5 * (squashed_action + 1.0) * (self.action_high - self.action_low)

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self._forward_stats(obs)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        raw_action = dist.rsample()
        squashed_action = torch.tanh(raw_action)
        action = self._scale_action(squashed_action)
        log_prob = dist.log_prob(raw_action) - torch.log(1.0 - squashed_action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self._forward_stats(obs)
        squashed_action = torch.tanh(mean)
        return self._scale_action(squashed_action)


class SACQNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: list[int]) -> None:
        super().__init__()
        self.net = _build_mlp(obs_dim + action_dim, hidden_sizes, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        obs = obs.view(obs.size(0), -1)
        action = action.view(action.size(0), -1)
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)
