
import torch
from torch import nn


def accuracy(preds, y):
    _, predicted = torch.max(preds, 1) # Find highest. 1 is the dimension
    return (predicted == y).sum().item() / y.size(0)


def mae(preds, y):
    return torch.mean(torch.abs(preds - y)).item()


def init_layer_weights(layer, kernel_initializer):

    # Initialize layer's weights using selected initializer
    if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
        if kernel_initializer == "glorot_uniform":
            nn.init.xavier_uniform_(layer.weight)
        elif kernel_initializer == "glorot_normal":
            nn.init.xavier_normal_(layer.weight)
        elif kernel_initializer == "he_uniform":
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        elif kernel_initializer == "he_normal":
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        else:
            raise ValueError(f"Initializer not supported: {kernel_initializer}")
        
        # If bias is present, initialize to zeros
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)