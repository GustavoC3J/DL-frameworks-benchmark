

import torch.nn as nn
import numpy as np

class MLPComplex(nn.Module):
    def __init__(self, groups, dropout, initial_units = 800, final_units = 160, layers_per_group = 2, activation = nn.ReLU(), input_dim = 784, kernel_initializer = "he_uniform"):
        """
        groups: Number of layer groups
        activation: Activation function
        dropout: Dropout rate
        layers_per_group: Number of hidden layers per group
        final_units: Last hidden layers will have these units
        input_dim: Number of features
        """
        super().__init__()

        # Define the number of units for each group
        units_per_group = np.linspace(initial_units, final_units, groups).astype(int)

        layers = nn.ModuleList()
        input_size = input_dim

        # Add the hidden layers
        for units in units_per_group:
            for _ in range(layers_per_group):
                linear = nn.Linear(input_size, units)
                self._init_weights(linear, kernel_initializer)
                layers.append(linear)
                layers.append(activation)
                layers.append(nn.Dropout(dropout))
                input_size = units  # Next layer input size is current output

        # Output layer
        layers.append(nn.Linear(final_units, 10))
        layers.append(nn.Softmax(dim=1))

        self.model = nn.Sequential(*layers)

    def _init_weights(self, layer, kernel_initializer):
        if isinstance(layer, nn.Linear):
            if kernel_initializer == "glorot_uniform":
                nn.init.xavier_uniform_(layer.weight)
            elif kernel_initializer == "glorot_normal":
                nn.init.xavier_normal_(layer.weight)
            elif kernel_initializer == "he_uniform":
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            elif kernel_initializer == "he_normal":
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            else:
                raise ValueError(f"Inicializador no soportado: {kernel_initializer}")
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.model(x)