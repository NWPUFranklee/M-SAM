# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from typing import Type
from .k_loralib import Linear as lora_Linear
from .k_loralib import MergedLinear, ConvLoRALinear1, ConvLoRALinear

class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class MLPBlock1(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
        lora_rank: int = 4,
    ) -> None:
        super().__init__()
        self.lin1 = lora_Linear(embedding_dim, mlp_dim, r=lora_rank)
        self.lin2 = lora_Linear(mlp_dim, embedding_dim, r=lora_rank)
        self.act = act()

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     return self.lin2(self.act(self.lin1(x)))
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        lin1x, lin1y = self.lin1(x, y)
        lin1x = self.act(lin1x)
        lin1y = self.act(lin1y)
        return self.lin2(lin1x, lin1y)
class MLPBlock2(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
        lora_rank: int = 4,
    ) -> None:
        super().__init__()
        self.lin1 = ConvLoRALinear1(embedding_dim, mlp_dim, r=lora_rank, conv_lora_expert_num=4)
        self.lin2 = ConvLoRALinear1(mlp_dim, embedding_dim, r=lora_rank, conv_lora_expert_num=4)
        self.act = act()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x1, y1 = self.lin1(x, y)
        x2, y2 = self.act(x1), self.act(y1)
        x3, y3 = self.lin2(x2, y2)
        return x3, y3 
