from typing import List, Optional, Union, Dict, Any
from dataclasses import dataclass, field
import json
import sys
import os
from neural_network import NeuralNetwork, activation_functions, loss_functions

@dataclass
class NetworkConfig:
    input_size: int = 64
    output_size: int = 6
    hidden_layers: List[int] = field(default_factory=lambda: [128, 64])
    activation_fns: List[str] = field(default_factory=lambda: ["relu", "relu"])
    output_activation: str = "softmax"
    dropout_rate: float = 0.0
    optimizer: str = "sgd"
    loss_function: str = "cross_entropy"
    weight_decay: float = 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NetworkConfig':
        valid_keys = cls.__annotations__.keys()
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        if "dropouts" in data and "dropout_rate" not in filtered_data:
            filtered_data["dropout_rate"] = float(data["dropouts"])
        
        return cls(**filtered_data)

class NetworkGenerator:
    @staticmethod
    def generate(config: NetworkConfig) -> NeuralNetwork:
        if config.loss_function not in loss_functions:
             raise ValueError(f"Invalid loss function: {config.loss_function}. Available: {list(loss_functions.keys())}")
        
        loss_fn = loss_functions[config.loss_function]
        model = NeuralNetwork(input_size=config.input_size, loss_function=loss_fn)
        if len(config.hidden_layers) != len(config.activation_fns):
            if len(config.activation_fns) == 1:
                config.activation_fns = config.activation_fns * len(config.hidden_layers)
            else:
                 raise ValueError(f"Mismatch between hidden_layers ({len(config.hidden_layers)}) and activation_fns ({len(config.activation_fns)}) counts.")

        for i, (size, act_name) in enumerate(zip(config.hidden_layers, config.activation_fns)):
            if act_name not in activation_functions:
                 raise ValueError(f"Invalid activation function: {act_name}")
            
            act_fn = activation_functions[act_name]
            model.add_layer(
                layer_size=size,
                activation=act_fn,
                dropout_rate=config.dropout_rate, 
                optimizer=config.optimizer,
                weight_decay=config.weight_decay
            )
            
        if config.output_activation not in activation_functions:
             raise ValueError(f"Invalid output activation: {config.output_activation}")
             
        model.add_layer(
            layer_size=config.output_size,
            activation=activation_functions[config.output_activation],
            optimizer=config.optimizer,
            weight_decay=config.weight_decay
        )
        
        return model

    @staticmethod
    def save_network(model: NeuralNetwork, filepath: str) -> None:
        try:
            model.save(filepath)
            print(f"[SUCCESS] Generated network saved to: {filepath}", file=sys.stdout)
        except Exception as e:
            print(f"[ERROR] Failed to save network to {filepath}: {e}", file=sys.stderr)
            raise

