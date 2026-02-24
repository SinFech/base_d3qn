from __future__ import annotations

from typing import Iterable

import torch
from torch.distributions import Normal
from torch import nn


def _build_mlp(
    input_dim: int,
    hidden_sizes: Iterable[int],
    output_dim: int,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = input_dim
    for size in hidden_sizes:
        layers.append(nn.Linear(current_dim, int(size)))
        layers.append(nn.Tanh())
        current_dim = int(size)
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


class ActorCriticMLP(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: list[int] | None = None,
    ) -> None:
        super().__init__()
        hidden = hidden_sizes or [256, 256]
        self.actor = _build_mlp(obs_dim, hidden, action_dim)
        self.critic = _build_mlp(obs_dim, hidden, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        obs = obs.view(obs.size(0), -1)
        logits = self.actor(obs)
        value = self.critic(obs).squeeze(-1)
        return logits, value


class GaussianActorCriticMLP(nn.Module):
    """Actor-critic network for continuous control with state-independent log-std."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: list[int] | None = None,
        init_log_std: float = -1.0,
    ) -> None:
        super().__init__()
        hidden = hidden_sizes or [256, 256]
        self.action_dim = int(action_dim)
        self.actor_mean = _build_mlp(obs_dim, hidden, self.action_dim)
        self.critic = _build_mlp(obs_dim, hidden, 1)
        self.log_std = nn.Parameter(torch.full((self.action_dim,), float(init_log_std)))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        obs = obs.view(obs.size(0), -1)
        mean = self.actor_mean(obs)
        value = self.critic(obs).squeeze(-1)
        return mean, value

    def distribution(self, obs: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        mean, value = self.forward(obs)
        std = torch.exp(self.log_std).unsqueeze(0).expand_as(mean)
        return Normal(mean, std), value
