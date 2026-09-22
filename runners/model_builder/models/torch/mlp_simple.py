

import torch.nn as nn

from utils.torch_utils import init_layer_weights

class MLPSimple(nn.Module):
    def __init__(self, activation, dropout):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 256),
            activation,
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            activation,
            nn.Dropout(dropout),

            nn.Linear(128, 10)
        )

        # Keras' default for Dense: torch draws kaiming uniform with a non-zero bias
        for layer in self.model:
            init_layer_weights(layer, "glorot_uniform")

    def forward(self, x):
        return self.model(x)