"""
The code is borrowed from Yuyang Li's codebase.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
# from pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction
from pointnet2_ops.pointnet2_modules import PointnetSAModule

from typing import List, Tuple, Sequence, Optional


class PointNet2(nn.Module):
    def __init__(self):

        super().__init__()

        # 6: xyz+xyz
        self.in_channel = 6

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=0.02,
                nsample=32,
                mlp=[self.in_channel, 32, 64],
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=64,
                radius=0.04,
                nsample=32,
                mlp=[64, 128, 128],
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=32,
                radius=0.08,
                nsample=16,
                mlp=[128, 256, 256]
            )
        )

        self.out_samples = 32

    def forward(self, pc: torch.Tensor) -> torch.Tensor:
        " pc: (B, N, 3+C) "
        xyz = pc[:, :, :3].contiguous()
        pc_extended = torch.cat([pc, pc], dim=-1)
        features = pc_extended.transpose(
            1, 2).contiguous()
        for module in self.SA_modules:
            xyz, features = module(xyz, features)

        return features  # (B, 256, 32)
