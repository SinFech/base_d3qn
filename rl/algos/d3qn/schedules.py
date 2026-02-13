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
        if step >= self.steps:
            return self.end
        if self.steps <= 0:
            return self.end
        # Linear decay: start -> end over steps (step is global env steps)
        frac = step / self.steps
        return self.start + frac * (self.end - self.start)
