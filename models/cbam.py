"""
models/cbam.py
==============
CBAM — Convolutional Block Attention Module
Paper: "CBAM: Convolutional Block Attention Module" (Woo et al., ECCV 2018)

Adds channel attention + spatial attention on top of any feature map.
Particularly effective for overhead/tilted-angle surveillance footage
where spatial focus on the arm/hand region distinguishes:
  - Touching vs Picking And Putting vs Picking And Returning

Usage in YOLO YAML:
    - [-1, 1, CBAM, [256]]   # 256 = number of channels
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    Channel Attention Module.
    Asks: WHICH feature channels are important?
    Uses both avg-pool and max-pool to capture channel statistics.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.mlp(self.avg_pool(x))
        max_ = self.mlp(self.max_pool(x))
        return self.sigmoid(avg + max_)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module.
    Asks: WHERE in the feature map is important?
    Key for overhead retail: focuses on hand/arm region to distinguish
    Touching from Picking.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel_size must be 3 or 7"
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Concat avg and max along channel dim
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg, max_], dim=1)
        return self.sigmoid(self.conv(attn))


class CBAM(nn.Module):
    """
    Full CBAM: Channel Attention → Spatial Attention.

    Args:
        channels: Number of input/output channels
        reduction: Channel reduction ratio for MLP (default: 16)
        kernel_size: Spatial attention kernel size (default: 7)

    Input/Output: same shape (B, C, H, W) — drop-in replacement

    Why this helps on CARR dataset:
        - Channel attention: focuses on relevant activation patterns
          for each of the 6 retail actions
        - Spatial attention: focuses on the body part that distinguishes
          actions (hands near shelf vs body turned away)
        - Overhead 45-90° tilt causes feature compression — attention
          helps the model compensate for this geometric distortion
    """

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: Channel attention — scale channels
        x = x * self.channel_attn(x)
        # Step 2: Spatial attention — scale spatial positions
        x = x * self.spatial_attn(x)
        return x
