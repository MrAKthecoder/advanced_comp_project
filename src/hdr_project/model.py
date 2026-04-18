from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SmallUNetHDR(nn.Module):
    """Lightweight U-Net for single-shot LDR-to-logHDR prediction."""

    def __init__(self, base: int = 32) -> None:
        super().__init__()
        self.enc1 = ConvBlock(3, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base * 2, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)

        self.head = nn.Conv2d(base, 3, kernel_size=1)
        self.softplus = nn.Softplus(beta=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))

        y2 = self.up2(x3)
        # Align shapes for odd-sized inputs before skip concatenation.
        if y2.shape[2:] != x2.shape[2:]:
            y2 = F.interpolate(y2, size=x2.shape[2:], mode="bilinear", align_corners=False)
        y2 = self.dec2(torch.cat([y2, x2], dim=1))
        y1 = self.up1(y2)
        # Align shapes for odd-sized inputs before skip concatenation.
        if y1.shape[2:] != x1.shape[2:]:
            y1 = F.interpolate(y1, size=x1.shape[2:], mode="bilinear", align_corners=False)
        y1 = self.dec1(torch.cat([y1, x1], dim=1))

        # Softplus keeps predictions positive for log-HDR values.
        return self.softplus(self.head(y1))
