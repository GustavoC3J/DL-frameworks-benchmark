

import torch.nn as nn

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

            nn.Linear(128, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)