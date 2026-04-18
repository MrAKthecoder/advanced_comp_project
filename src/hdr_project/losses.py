from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HDRReconstructionLoss(nn.Module):
    """Combination of pixel L1 and gradient L1 losses.

    Gradient term helps preserve structure and edges in HDR reconstruction.
    """

    def __init__(self, grad_weight: float = 0.2) -> None:
        super().__init__()
        self.grad_weight = grad_weight

    @staticmethod
    def _gradient_map(x: torch.Tensor) -> torch.Tensor:
        gx = x[:, :, :, 1:] - x[:, :, :, :-1]
        gy = x[:, :, 1:, :] - x[:, :, :-1, :]
        gx = F.pad(gx, (0, 1, 0, 0))
        gy = F.pad(gy, (0, 0, 0, 1))
        return torch.cat([gx, gy], dim=1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1 = F.l1_loss(pred, target)
        grad = F.l1_loss(self._gradient_map(pred), self._gradient_map(target))
        return l1 + self.grad_weight * grad
