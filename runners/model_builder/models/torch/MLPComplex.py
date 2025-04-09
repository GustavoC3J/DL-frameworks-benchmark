

import torch.nn as nn

class MLPComplex(nn.Module):
    def __init__(self, activation, dropout):
        super().__init__()

        hidden_layers = 15
        final_units = 64  # Last hidden layers will have these units
        layers_per_group = 3

        # Calculate initial units based on number of hidden layers
        groups = (hidden_layers + layers_per_group - 1) // layers_per_group
        units = final_units * (2 ** (groups - 1))  # Starting units

        layers = nn.ModuleList()
        input_dim = 784  # Inputs

        # Add the hidden layers
        for i in range(1, hidden_layers + 1):
            layers.append(nn.Linear(input_dim, units))
            layers.append(activation)
            layers.append(nn.Dropout(dropout))
            input_dim = units  # Next layer input size is current output

            # Halve the number of units for the next group
            if i % layers_per_group == 0:
                units //= 2

        # Output layer
        layers.append(nn.Linear(final_units, 10))
        layers.append(nn.Softmax(dim=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)