import torch
import torch.nn as nn
from typing import List


class DynamicChessNet(nn.Module):
    def __init__(self, input_size: int, hidden_layers: List[int], output_size: int, activation_fn: str = "relu"):
        super().__init__()
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(negative_slope=0.01),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh()
        }
        act_layer = activations.get(activation_fn.lower(), nn.ReLU())

        layers = []
        current_input_dim = input_size
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(current_input_dim, hidden_dim))
            layers.append(act_layer)
            current_input_dim = hidden_dim
            
        layers.append(nn.Linear(current_input_dim, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)