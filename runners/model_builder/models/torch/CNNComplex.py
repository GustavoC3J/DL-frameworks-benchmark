
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if stride > 1 or in_channels != out_channels:
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
    
    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        
        if self.downsample:
            residual = self.downsample(residual)
        
        x += residual
        x = F.relu(x)

        return x



class CNNComplex(nn.Module):
    def __init__(self, n_blocks):
        super().__init__()
        self.in_channels = 16
        
        layers = nn.ModuleList()
        layers.append(nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(16))
        layers.append(nn.ReLU())

        # Each stage is composed of n blocks whose convolutions use the corresponding filters
        filters = [16, 32, 64]
        for stage, out_channels in enumerate(filters):
            for i in range(n_blocks):
                # If it is the first block of the stage, a subsampling is made
                stride = 2 if stage > 0 and i == 0 else 1

                layers.append(Block(self.in_channels, out_channels, stride=stride))
                self.in_channels = out_channels

        # Flatten and perform final prediction
        layers.append(nn.AdaptiveAvgPool2d(1)) # shape [batch_size, channels, 1, 1]
        layers.append(nn.Flatten()) # shape [batch_size, channels]

        layers.append(nn.Linear(self.in_channels, 10))
        layers.append(nn.Softmax(dim=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Switch to (batch_size, channels, height, width)
        x = x.permute(0, 3, 2, 1)

        return self.model(x)
