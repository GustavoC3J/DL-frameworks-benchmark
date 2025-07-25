
import torch.nn as nn
import torch.nn.functional as F

class CNNSimple(nn.Module):
    def __init__(self, activation, dropout):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding='same'),
            activation,
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(dropout),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same'),
            activation,
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(dropout),

            nn.Flatten(),

            # 32 * 8 * 8 is the output size after convolutions and max pooling
            nn.Linear(32 * 8 * 8, 128),
            activation,
            nn.Dropout(dropout),

            nn.Linear(128, 128),
            activation,
            nn.Dropout(dropout),

            # Output layer
            nn.Linear(128, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # Switch to (batch_size, channels, height, width)
        x = x.permute(0, 3, 2, 1)

        return self.model(x)