"""
MLP — многослойный перцептрон.

Адаптировано из Kinesis (SMPLSim, PyTorch-RL).
"""

import torch
import torch.nn as nn
from typing import Union, Tuple


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Union[Tuple[int, ...], list],
        activation: str = "tanh",
    ):
        super().__init__()
        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu
        elif activation == "sigmoid":
            self.activation = torch.sigmoid
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.out_dim = hidden_dims[-1]
        self.affine_layers = nn.ModuleList()
        last_dim = input_dim
        for nh in hidden_dims:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        return x
