from __future__ import annotations

import torch.nn.functional as F
import torch


def mse_q_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(predicted, target)
