from __future__ import annotations

from collections import namedtuple
from typing import List
import random

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.memory: List[Transition] = []
        self.position = 0

    def push(self, *args) -> None:
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)
