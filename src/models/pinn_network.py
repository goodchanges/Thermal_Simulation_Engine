import torch
import torch.nn as nn

class ThermalSurrogateModel(nn.Module):
    def __init__(self, num_hidden_layers=5, neurons_per_layer=64):
        super(ThermalSurrogateModel, self).__init__()
        
        # Input layer: x, y, z coordinates + time (t)
        layers = [nn.Linear(4, neurons_per_layer), nn.Tanh()]
        
        # Hidden layers for deep feature extraction
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(nn.Tanh())
            
        # Output layer: Temperature (or structural displacement)
        layers.append(nn.Linear(neurons_per_layer, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        # Strict deterministic forward pass
        return self.network(x)