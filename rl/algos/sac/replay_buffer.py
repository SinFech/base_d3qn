from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device: str) -> None:
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)

        self.obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: float,
    ) -> None:
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        indices = np.random.randint(0, self.size, size=int(batch_size))
        return {
            "obs": torch.as_tensor(self.obs[indices], device=self.device, dtype=torch.float32),
            "actions": torch.as_tensor(self.actions[indices], device=self.device, dtype=torch.float32),
            "rewards": torch.as_tensor(self.rewards[indices], device=self.device, dtype=torch.float32),
            "next_obs": torch.as_tensor(self.next_obs[indices], device=self.device, dtype=torch.float32),
            "dones": torch.as_tensor(self.dones[indices], device=self.device, dtype=torch.float32),
        }

    def __len__(self) -> int:
        return self.size
