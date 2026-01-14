from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EpsilonSchedule:
    start: float
    end: float
    steps: int

    def value(self, step: int, training: bool = True) -> float:
        if not training:
            return self.end
        if step > self.steps:
            return self.end
        return self.start
